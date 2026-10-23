from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from annaops.agents.receiver_advisor import ReceiverAdvisorAgent
from annaops.agents.safety import SafetyAgent
from annaops.config import Settings, settings
from annaops.db import Database
from annaops.schemas import AgentDecision, DonationStatus, ExtractedOffer, SafetyVerdict


class InvalidTransition(ValueError):
    pass


class DispatchAgent:
    """Enforces approval, dispatch, retry, escalation, and delivery-confirmation gates."""

    def __init__(self, database: Database, config: Settings = settings):
        self.database = database
        self.config = config
        self.safety_agent = SafetyAgent(config)
        self.receiver_advisor = ReceiverAdvisorAgent(database, config)

    def approve(self, donation_id: str, actor: str, reason: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        self._require(donation, {DonationStatus.AWAITING_DONOR_APPROVAL})
        invalidated = self._invalidate_if_unsafe(donation)
        if invalidated:
            return invalidated
        ngo_id = donation["selected_ngo_id"]
        if not ngo_id:
            raise InvalidTransition("No receiver is selected")
        attempt_number = donation["attempt_count"] + 1
        attempted = [*donation["attempted_ngo_ids"], ngo_id]
        updated = self.database.update_donation(
            donation_id, donation["status"], DonationStatus.AWAITING_NGO_RESPONSE,
            "DONOR_APPROVED_AND_DISPATCHED", actor,
            reason or "Donor explicitly approved the proposed receiver",
            {"ngo_id": ngo_id, "attempt_number": attempt_number, "simulated": True},
            donor_approved=True, attempt_count=attempt_number, attempted_ngo_ids_json=attempted,
        )
        self.database.add_approval(donation_id, "APPROVED", actor, reason)
        self.database.create_dispatch(donation_id, ngo_id, attempt_number)
        result = self.database.get_donation(donation_id)
        self._record_simple(
            result, "dispatch_agent", "dispatch", "Validate approval and create one simulated dispatch",
            "dispatch_to_receiver", {"ngo_id": ngo_id, "attempt_number": attempt_number},
            ["Donor approval was explicit", "Safety was revalidated before dispatch"],
            ["revalidate_safety", "create_simulated_dispatch"], started,
        )
        return self.database.get_donation(donation_id)

    def reject_proposal(self, donation_id: str, actor: str, reason: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        self._require(donation, {DonationStatus.AWAITING_DONOR_APPROVAL})
        updated = self.database.update_donation(
            donation_id, donation["status"], DonationStatus.DONOR_REJECTED,
            "DONOR_REJECTED_PROPOSAL", actor, reason or "Donor rejected the proposed dispatch",
            {"selected_ngo_id": donation["selected_ngo_id"]}, donor_approved=False,
        )
        self.database.add_approval(donation_id, "REJECTED", actor, reason)
        self._record_simple(
            updated, "dispatch_agent", "approval", "Respect the donor approval gate",
            "stop_after_donor_rejection", {}, ["Donor explicitly rejected the proposal"],
            ["persist_approval_decision"], started,
        )
        return updated

    def respond(self, donation_id: str, response: str, actor: str, reason: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        self._require(donation, {DonationStatus.AWAITING_NGO_RESPONSE})
        invalidated = self._invalidate_if_unsafe(donation)
        if invalidated:
            return invalidated
        attempt = self.database.resolve_dispatch(donation_id, response, reason)
        if response == "accept":
            result = self.database.update_donation(
                donation_id, donation["status"], DonationStatus.ACCEPTED,
                "NGO_ACCEPTED", actor,
                reason or "Synthetic receiver accepted the simulated dispatch",
                {"ngo_id": attempt["ngo_id"], "simulated": True, "delivery_confirmed": False},
                final_outcome_json={
                    "arrangement": "accepted", "delivery": "unconfirmed", "synthetic": True,
                    "note": "Acceptance confirms only the simulated arrangement."
                },
            )
            self._record_simple(
                result, "dispatch_agent", "response", "Interpret a persisted simulated receiver response",
                "await_delivery_confirmation", {"ngo_id": attempt["ngo_id"]},
                ["Receiver accepted the arrangement", "Acceptance is not delivery confirmation"],
                ["read_dispatch_state", "persist_receiver_response"], started,
            )
            return result
        return self._replan_or_escalate(donation, response, actor, reason)

    def confirm_delivery(self, donation_id: str, actor: str, verification_note: str) -> dict[str, Any]:
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        self._require(donation, {DonationStatus.ACCEPTED})
        timestamp = datetime.now(UTC).isoformat()
        result = self.database.update_donation(
            donation_id, donation["status"], DonationStatus.COMPLETED,
            "DELIVERY_CONFIRMED", actor, verification_note,
            {"verification_note": verification_note, "synthetic": True},
            delivery_confirmed_at=timestamp,
            final_outcome_json={
                "arrangement": "accepted", "delivery": "confirmed_in_demo", "confirmed_at": timestamp,
                "verification_note": verification_note, "synthetic": True,
            },
        )
        self._record_simple(
            result, "dispatch_agent", "completion", "Keep acceptance separate from verified demo delivery",
            "complete_demo_delivery", {}, ["A separate coordinator confirmation was supplied"],
            ["persist_delivery_confirmation"], started,
        )
        return result

    def escalate(self, donation_id: str, actor: str, reason: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        allowed = {
            DonationStatus.NEEDS_CLARIFICATION, DonationStatus.REQUIRES_HUMAN_REVIEW,
            DonationStatus.AWAITING_DONOR_APPROVAL, DonationStatus.AWAITING_NGO_RESPONSE,
            DonationStatus.REPLANNING,
        }
        self._require(donation, allowed)
        result = self.database.update_donation(
            donation_id, donation["status"], DonationStatus.ESCALATED,
            "HUMAN_ESCALATION", actor, reason or "Escalated for coordinator review",
            {"attempt_count": donation["attempt_count"]},
        )
        self._record_simple(
            result, "dispatch_agent", "escalation", "Persist an explicit human escalation",
            "escalate", {}, [reason or "Coordinator requested escalation"],
            ["read_dispatch_state", "persist_escalation"], started,
        )
        return result

    def _replan_or_escalate(self, donation: dict[str, Any], response: str, actor: str,
                            reason: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        donation_id = donation["id"]
        remaining_minutes = None
        parsed = ExtractedOffer.model_validate(donation["parsed"])
        if parsed.pickup_deadline:
            remaining_minutes = (parsed.pickup_deadline - datetime.now(UTC)).total_seconds() / 60
        should_escalate = (
            donation["attempt_count"] >= self.config.max_dispatch_attempts
            or (remaining_minutes is not None and remaining_minutes < self.config.escalation_window_minutes)
        )
        failure_event = "NGO_DECLINED" if response == "decline" else "NGO_TIMEOUT"
        if should_escalate:
            result = self.database.update_donation(
                donation_id, donation["status"], DonationStatus.ESCALATED,
                failure_event + "_ESCALATED", actor,
                reason or f"Synthetic receiver {response}; retry or time threshold reached",
                {"attempt_count": donation["attempt_count"], "remaining_minutes": remaining_minutes},
            )
            self._record_simple(
                result, "dispatch_agent", "replanning", "Choose between bounded retry and escalation",
                "escalate", {"remaining_minutes": remaining_minutes},
                ["Configured retry or collection-window threshold was reached"],
                ["read_dispatch_state", "inspect_collection_window"], started,
            )
            return result

        advice = self.receiver_advisor.recommend(donation, parsed, set(donation["attempted_ngo_ids"]))
        match = advice.match
        self.database.add_match_run(donation_id, match.model_dump(mode="json"))
        if not advice.selected_ngo_id:
            result = self.database.update_donation(
                donation_id, donation["status"], DonationStatus.ESCALATED,
                failure_event + "_NO_ALTERNATIVE", actor,
                reason or "No eligible untried receiver remains",
                {"attempted_ngo_ids": donation["attempted_ngo_ids"]}, match_json=match.model_dump(mode="json"),
            )
        else:
            next_ngo = advice.selected_ngo_id
            result = self.database.update_donation(
                donation_id, donation["status"], DonationStatus.AWAITING_DONOR_APPROVAL,
                failure_event + "_REPLANNED", actor,
                reason or f"Synthetic receiver {response}; an alternative requires fresh donor approval",
                {"previous_ngo_id": donation["selected_ngo_id"], "next_ngo_id": next_ngo},
                match_json=match.model_dump(mode="json"), selected_ngo_id=next_ngo, donor_approved=False,
            )
        trace = advice.trace
        self.database.record_agent_execution(
            donation_id, agent_type="receiver_advisor_agent", workflow_stage="replanning",
            objective="Inspect previous attempts and live alternatives after a failed receiver response",
            execution_mode=trace["mode"], provider=trace["provider"], model=trace["model"],
            input_summary={"response": response, "attempted_ngo_ids": donation["attempted_ngo_ids"],
                           "remaining_minutes": remaining_minutes},
            structured_output=trace["structured_output"], tools=trace["tools"],
            selected_action=trace["selected_action"], resulting_status=result["status"],
            error=trace["error"], retry_count=trace["retry_count"], duration_ms=trace["duration_ms"],
            started_at=trace.get("started_at"),
        )
        return self.database.get_donation(donation_id)

    def _invalidate_if_unsafe(self, donation: dict[str, Any]) -> dict[str, Any] | None:
        assessment = self.safety_agent.assess(ExtractedOffer.model_validate(donation["parsed"]))
        if assessment.verdict == SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING:
            return None
        return self.database.update_donation(
            donation["id"], donation["status"], DonationStatus.BLOCKED,
            "SAFETY_INVALIDATED", "system", "Safety was rechecked before dispatch action",
            {"assessment": assessment.model_dump(mode="json")}, safety_json=assessment.model_dump(mode="json"),
        )

    @staticmethod
    def _require(donation: dict[str, Any], allowed: set[DonationStatus]) -> None:
        if donation["status"] not in {item.value for item in allowed}:
            expected = ", ".join(sorted(item.value for item in allowed))
            raise InvalidTransition(f"Action requires status {expected}; current status is {donation['status']}")

    def _record_simple(
        self,
        donation: dict[str, Any],
        agent_type: str,
        stage: str,
        objective: str,
        action: str,
        action_arguments: dict[str, Any],
        evidence: list[str],
        tool_names: list[str],
        started: float,
    ) -> None:
        decision = AgentDecision(
            current_task=objective,
            observed_facts=[f"Persisted status is {donation['status']}"],
            proposed_action=action,
            action_arguments=action_arguments,
            evidence=evidence,
            expected_outcome=f"Workflow state remains durably persisted as {donation['status']}",
            human_input_required=donation["status"] in {
                DonationStatus.AWAITING_DONOR_APPROVAL.value,
                DonationStatus.AWAITING_NGO_RESPONSE.value,
                DonationStatus.ACCEPTED.value,
                DonationStatus.ESCALATED.value,
            },
            task_complete=True,
        )
        self.database.record_agent_execution(
            donation["id"], agent_type=agent_type, workflow_stage=stage, objective=objective,
            execution_mode="deterministic", input_summary={"status": donation["status"]},
            structured_output=decision.model_dump(mode="json"),
            tools=[{"sequence": index, "name": name, "arguments": {}, "purpose": objective,
                    "result": {"resulting_status": donation["status"]}}
                   for index, name in enumerate(tool_names, 1)],
            selected_action=action, resulting_status=donation["status"],
            duration_ms=round((time.perf_counter() - started) * 1000),
        )


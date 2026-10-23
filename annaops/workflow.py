from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from annaops.agents.intake import IntakeAgent
from annaops.agents.coordinator import WorkflowCoordinatorAgent
from annaops.agents.receiver_advisor import ReceiverAdvisorAgent
from annaops.agents.safety import SafetyAgent
from annaops.config import Settings, settings
from annaops.db import Database
from annaops.schemas import DonationStatus, ExtractedOffer, SafetyVerdict


class WorkflowState(TypedDict, total=False):
    donation_id: str
    original_message: str
    parsed: dict[str, Any]
    safety: dict[str, Any]
    match: dict[str, Any]
    outcome: str
    skip_intake: bool
    next_operation: str
    coordinator_steps: int


class DonationWorkflow:
    """LangGraph topology for intake, completeness, safety, and matching.

    Human approval and external-response waits are durable API-driven transitions handled by DispatchAgent.
    """

    def __init__(self, database: Database, config: Settings = settings):
        self.database = database
        self.config = config
        self.intake_agent = IntakeAgent(config)
        self.safety_agent = SafetyAgent(config)
        self.receiver_advisor = ReceiverAdvisorAgent(database, config)
        self.coordinator = WorkflowCoordinatorAgent(database)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("intake_agent", self._intake_node)
        graph.add_node("safety_agent", self._safety_node)
        graph.add_node("match_route_agent", self._match_node)
        graph.add_edge(START, "coordinator")
        graph.add_conditional_edges(
            "coordinator", self._after_coordinator,
            {"intake": "intake_agent", "safety": "safety_agent", "match": "match_route_agent", "stop": END},
        )
        graph.add_edge("intake_agent", "coordinator")
        graph.add_edge("safety_agent", "coordinator")
        graph.add_edge("match_route_agent", "coordinator")
        return graph.compile()

    def _coordinator_node(self, state: WorkflowState) -> WorkflowState:
        steps = state.get("coordinator_steps", 0) + 1
        if steps > 6:
            raise RuntimeError("Coordinator iteration limit reached")
        operation = self.coordinator.coordinate(state["donation_id"])
        return {**state, "next_operation": operation, "coordinator_steps": steps}

    @staticmethod
    def _after_coordinator(state: WorkflowState) -> Literal["intake", "safety", "match", "stop"]:
        return {
            "run_intake": "intake",
            "run_safety": "safety",
            "run_matching": "match",
        }.get(state["next_operation"], "stop")

    def process_new(self, donation_id: str, message: str) -> dict[str, Any]:
        self.graph.invoke({"donation_id": donation_id, "original_message": message})
        return self.database.get_donation(donation_id)

    def process_clarification(self, donation_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        donation = self.database.get_donation(donation_id)
        if donation["status"] not in {
            DonationStatus.NEEDS_CLARIFICATION.value,
            DonationStatus.REQUIRES_HUMAN_REVIEW.value,
        }:
            raise ValueError(f"Clarification is not allowed in status {donation['status']}")
        current = ExtractedOffer.model_validate(donation["parsed"])
        current = self.intake_agent.upgrade_legacy_food_items(current, donation["original_message"])
        reference_at = datetime.fromisoformat(donation["created_at"])
        merged = self.intake_agent.merge_clarification(current, fields, reference_at=reference_at)
        self.database.update_donation(
            donation_id, donation["status"], DonationStatus.INTAKE_PROCESSING,
            "CLARIFICATION_SUBMITTED", "donor", "Donor supplied clarification fields",
            {"field_names": sorted(fields)}, parsed_json=merged.model_dump(mode="json"),
        )
        self.graph.invoke({
            "donation_id": donation_id, "original_message": donation["original_message"],
            "parsed": merged.model_dump(mode="json"), "skip_intake": True,
        })
        return self.database.get_donation(donation_id)

    def _intake_node(self, state: WorkflowState) -> WorkflowState:
        donation = self.database.get_donation(state["donation_id"])
        if state.get("skip_intake") and state.get("parsed"):
            started = time.perf_counter()
            parsed = ExtractedOffer.model_validate(state["parsed"])
            trace = {
                "mode": "deterministic", "provider": None, "model": None,
                "duration_ms": round((time.perf_counter() - started) * 1000), "retry_count": 0,
                "error": None, "structured_output": parsed.model_dump(mode="json"), "tools": [],
                "selected_action": "request_clarification" if parsed.missing_fields or parsed.contradictions else "submit_for_safety",
            }
        else:
            parsed, trace = self.intake_agent.extract_with_trace(state["original_message"])
        needs_clarification = bool(
            parsed.missing_fields
            or parsed.contradictions
            or parsed.extraction_source == "llm_error"
        )
        new_status = DonationStatus.NEEDS_CLARIFICATION if needs_clarification else DonationStatus.PENDING_SAFETY_REVIEW
        updated = self.database.update_donation(
            state["donation_id"], donation["status"], new_status,
            "INTAKE_CLARIFICATION_REQUIRED" if needs_clarification else "INTAKE_PARSED",
            "intake_agent",
            "Critical information is missing or contradictory" if needs_clarification else "Structured extraction validated",
            {"missing_fields": parsed.missing_fields, "contradictions": parsed.contradictions,
             "source": parsed.extraction_source},
            parsed_json=parsed.model_dump(mode="json"),
        )
        self.database.record_agent_execution(
            state["donation_id"], agent_type="intake_agent", workflow_stage="intake",
            objective="Extract explicit donor facts and identify only blocking clarification needs",
            execution_mode=trace["mode"], provider=trace["provider"], model=trace["model"],
            input_summary={"message_length": len(state["original_message"]), "existing_record": bool(state.get("skip_intake"))},
            structured_output=trace["structured_output"], tools=trace["tools"],
            selected_action=trace["selected_action"], resulting_status=updated["status"],
            error=trace["error"], retry_count=trace["retry_count"], duration_ms=trace["duration_ms"],
        )
        return {**state, "parsed": parsed.model_dump(mode="json"), "outcome": updated["status"]}

    def _safety_node(self, state: WorkflowState) -> WorkflowState:
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        donation = self.database.get_donation(state["donation_id"])
        assessment = self.safety_agent.assess(ExtractedOffer.model_validate(state["parsed"]))
        status_by_verdict = {
            SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING: DonationStatus.MATCHING,
            SafetyVerdict.NEEDS_CLARIFICATION: DonationStatus.NEEDS_CLARIFICATION,
            SafetyVerdict.BLOCKED: DonationStatus.BLOCKED,
            SafetyVerdict.REQUIRES_HUMAN_REVIEW: DonationStatus.REQUIRES_HUMAN_REVIEW,
        }
        new_status = status_by_verdict[assessment.verdict]
        self.database.add_safety_assessment(state["donation_id"], assessment.model_dump(mode="json"))
        updated = self.database.update_donation(
            state["donation_id"], donation["status"], new_status,
            "SAFETY_ASSESSED", "safety_agent", "; ".join(assessment.reasons),
            {"verdict": assessment.verdict, "rule_version": assessment.rule_version},
            safety_json=assessment.model_dump(mode="json"),
        )
        self.database.record_agent_execution(
            state["donation_id"], agent_type="safety_agent", workflow_stage="safety",
            objective="Apply authoritative versioned food-safety rules",
            execution_mode="deterministic", input_summary={"parsed_fields": sorted(state["parsed"].keys())},
            structured_output=assessment.model_dump(mode="json"),
            tools=[{"sequence": 1, "name": "run_versioned_safety_rules", "arguments": {"rule_version": assessment.rule_version},
                    "purpose": "Produce the authoritative safety verdict", "result": assessment.model_dump(mode="json")}],
            selected_action="run_matching" if new_status == DonationStatus.MATCHING else "durable_pause",
            resulting_status=updated["status"], duration_ms=round((time.perf_counter() - started) * 1000),
            started_at=started_at,
        )
        return {**state, "safety": assessment.model_dump(mode="json"), "outcome": new_status}

    def _match_node(self, state: WorkflowState) -> WorkflowState:
        donation = self.database.get_donation(state["donation_id"])
        advice = self.receiver_advisor.recommend(donation, ExtractedOffer.model_validate(state["parsed"]))
        result = advice.match
        self.database.add_match_run(state["donation_id"], result.model_dump(mode="json"))
        if not advice.selected_ngo_id:
            new_status = DonationStatus.ESCALATED
            selected = None
            reason = "No eligible receiver was found"
        else:
            new_status = DonationStatus.AWAITING_DONOR_APPROVAL
            selected = advice.selected_ngo_id
            reason = "Agent recommendation is ready for explicit donor approval"
        updated = self.database.update_donation(
            state["donation_id"], donation["status"], new_status,
            "MATCHING_COMPLETED", "match_route_agent", reason,
            {"candidate_count": len(result.candidates), "selected_ngo_id": selected},
            match_json=result.model_dump(mode="json"), selected_ngo_id=selected,
        )
        trace = advice.trace
        self.database.record_agent_execution(
            state["donation_id"], agent_type="receiver_advisor_agent", workflow_stage="matching",
            objective="Use live receiver tools and recommend only an authoritatively eligible receiver",
            execution_mode=trace["mode"], provider=trace["provider"], model=trace["model"],
            input_summary={"food_category": state["parsed"].get("food_category"), "attempted_ngo_ids": []},
            structured_output=trace["structured_output"], tools=trace["tools"],
            selected_action=trace["selected_action"], resulting_status=updated["status"],
            error=trace["error"], retry_count=trace["retry_count"], duration_ms=trace["duration_ms"],
            started_at=trace.get("started_at"),
        )
        return {**state, "match": result.model_dump(mode="json"), "outcome": new_status}


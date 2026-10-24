from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from annaops.agents.coordinator import WorkflowCoordinatorAgent
from annaops.agents.intake import IntakeAgent
from annaops.agents.receiver_advisor import ReceiverAdvisorAgent
from annaops.agents.runtime import AgentProtocolError
from annaops.agents.safety import SafetyAgent
from annaops.config import CANONICAL_MODEL, Settings, settings
from annaops.db import BudgetExceeded, Database
from annaops.llm import LLMError, StructuredLLMClient
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

    Uses the configured LLM provider for designated LLM operations. Propagates failures
    safely without silent fallback.
    """

    def __init__(self, database: Database, config: Settings = settings, client: StructuredLLMClient | Any | None = None):
        self.database = database
        self.config = config
        self.intake_agent = IntakeAgent(config, client=client, database=database)
        self.safety_agent = SafetyAgent(config)
        self.receiver_advisor = ReceiverAdvisorAgent(database, config, client=client)
        self.coordinator = WorkflowCoordinatorAgent(database, config, client=client)
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
        graph.add_conditional_edges(
            "intake_agent", self._after_intake,
            {"safety": "safety_agent", "stop": END},
        )
        graph.add_conditional_edges(
            "safety_agent", self._after_safety,
            {"match": "match_route_agent", "stop": END},
        )
        graph.add_edge("match_route_agent", END)
        return graph.compile()

    @staticmethod
    def _after_intake(state: WorkflowState) -> Literal["safety", "stop"]:
        return "safety" if state.get("outcome") == DonationStatus.PENDING_SAFETY_REVIEW.value else "stop"

    @staticmethod
    def _after_safety(state: WorkflowState) -> Literal["match", "stop"]:
        return "match" if state.get("outcome") == DonationStatus.MATCHING.value else "stop"

    def _coordinator_node(self, state: WorkflowState) -> WorkflowState:
        steps = state.get("coordinator_steps", 0) + 1
        if steps > 8:
            raise RuntimeError("Coordinator iteration limit reached")
        operation = self.coordinator.coordinate(state["donation_id"])
        return {**state, "next_operation": operation, "coordinator_steps": steps}

    @staticmethod
    def _after_coordinator(state: WorkflowState) -> Literal["intake", "safety", "match", "stop"]:
        return {
            "run_intake": "intake",
            "run_safety": "safety",
            "run_matching": "match",
        }.get(state.get("next_operation", "durable_pause"), "stop")

    def process_new(self, donation_id: str, message: str) -> dict[str, Any]:
        self.graph.invoke({"donation_id": donation_id, "original_message": message})
        return self.database.get_donation(donation_id)

    def retry_processing(self, donation_id: str) -> dict[str, Any]:
        """Authorized retry of a donation in ERROR status once provider is available."""
        donation = self.database.get_donation(donation_id)
        if donation["status"] != DonationStatus.ERROR.value:
            raise ValueError(f"Only donations in ERROR status can be retried; current status is {donation['status']}")

        parsed = donation.get("parsed")
        safety = donation.get("safety")

        if parsed and safety and safety.get("verdict") == SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING.value:
            self.database.update_donation(
                donation_id, DonationStatus.ERROR, DonationStatus.MATCHING,
                "RETRY_INITIATED", "coordinator", "Authorised retry resuming at matching stage",
            )
            self.graph.invoke({
                "donation_id": donation_id,
                "original_message": donation["original_message"],
                "parsed": parsed,
                "safety": safety,
                "skip_intake": True,
            })
        else:
            self.database.update_donation(
                donation_id, DonationStatus.ERROR, DonationStatus.INTAKE_PROCESSING,
                "RETRY_INITIATED", "coordinator", "Authorised retry resuming at intake stage",
            )
            self.graph.invoke({"donation_id": donation_id, "original_message": donation["original_message"]})

        return self.database.get_donation(donation_id)

    def submit_clarification(self, donation_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        donation = self.database.get_donation(donation_id)
        if donation["status"] != DonationStatus.NEEDS_CLARIFICATION.value:
            raise ValueError(f"Donation {donation_id} does not require clarification")
        current = ExtractedOffer.model_validate(donation["parsed"])
        reference = datetime.fromisoformat(donation["created_at"].replace("Z", "+00:00"))
        merged = self.intake_agent.merge_clarification(current, fields, reference_at=reference)
        self.database.update_donation(
            donation_id, donation["status"], DonationStatus.INTAKE_PROCESSING,
            "CLARIFICATION_SUBMITTED", "donor", "Donor supplied clarification fields",
            {"field_names": sorted(fields)}, parsed_json=merged.model_dump(mode="json"),
        )
        state: WorkflowState = {
            "donation_id": donation_id,
            "original_message": donation["original_message"],
            "parsed": merged.model_dump(mode="json"),
            "skip_intake": True,
        }
        try:
            # Clarification has already been validated and persisted. Resume through the
            # existing authoritative nodes directly instead of spending two model calls
            # asking the coordinator to rediscover the deterministic next operation.
            state = self._intake_node(state)
            if self._after_intake(state) == "safety":
                state = self._safety_node(state)
                if self._after_safety(state) == "match":
                    self._match_node(state)
        except BudgetExceeded as exc:
            current = self.database.get_donation(donation_id)
            if current["status"] != DonationStatus.ERROR.value:
                self.database.update_donation(
                    donation_id,
                    current["status"],
                    DonationStatus.ERROR,
                    "WORKFLOW_ERROR",
                    "system",
                    "Clarification was saved, but processing stopped because the configured LLM budget was exhausted",
                    {"error_type": type(exc).__name__, "stage": current["status"]},
                    error_message=str(exc),
                )
            raise
        return self.database.get_donation(donation_id)

    def review_safety(self, donation_id: str, decision: str, actor: str, reason: str) -> dict[str, Any]:
        donation = self.database.get_donation(donation_id)
        if donation["status"] != DonationStatus.REQUIRES_HUMAN_REVIEW.value:
            raise ValueError(f"Donation {donation_id} is not awaiting safety review")
        parsed = ExtractedOffer.model_validate(donation.get("parsed") or {})
        if decision == "reject":
            return self.database.update_donation(
                donation_id, donation["status"], DonationStatus.BLOCKED,
                "SAFETY_REVIEW_REJECTED", actor, reason,
                {"decision": decision, "rule_version": self.config.safety_rule_version},
            )
        if decision == "request_clarification":
            missing = list(dict.fromkeys([*parsed.missing_fields, "storage_conditions"]))
            questions = list(dict.fromkeys([
                *parsed.clarification_questions,
                "Please provide a recognized storage condition and any verification details.",
            ]))
            previous_storage = parsed.storage_conditions
            parsed = parsed.model_copy(update={
                "storage_conditions": None,
                "missing_fields": missing,
                "clarification_questions": questions,
            })
            return self.database.update_donation(
                donation_id, donation["status"], DonationStatus.NEEDS_CLARIFICATION,
                "SAFETY_REVIEW_CLARIFICATION_REQUESTED", actor, reason,
                {
                    "decision": decision,
                    "missing_fields": missing,
                    "invalid_fields": ["storage_conditions"],
                    "previous_storage_condition": previous_storage,
                },
                parsed_json=parsed.model_dump(mode="json"),
            )
        assessment = self.safety_agent.assess(parsed)
        self.database.add_safety_assessment(donation_id, assessment.model_dump(mode="json"))
        if assessment.verdict != SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING:
            raise ValueError(
                "Safety review approval rejected: revalidation remains "
                f"{assessment.verdict.value}. Resolve the safety findings first."
            )
        updated = self.database.update_donation(
            donation_id, donation["status"], DonationStatus.MATCHING,
            "SAFETY_REVIEW_APPROVED", actor, reason,
            {"decision": decision, "rule_version": assessment.rule_version},
            safety_json=assessment.model_dump(mode="json"),
        )
        self.graph.invoke({
            "donation_id": donation_id,
            "original_message": donation["original_message"],
            "parsed": parsed.model_dump(mode="json"),
            "safety": assessment.model_dump(mode="json"),
            "skip_intake": True,
        })
        return self.database.get_donation(donation_id)

    def _intake_node(self, state: WorkflowState) -> WorkflowState:
        donation = self.database.get_donation(state["donation_id"])
        if state.get("skip_intake") and state.get("parsed"):
            started = time.perf_counter()
            parsed = ExtractedOffer.model_validate(state["parsed"])
            trace = {
                "mode": "deterministic", "provider": "authoritative_schema", "model": None,
                "duration_ms": round((time.perf_counter() - started) * 1000), "retry_count": 0,
                "error": None, "structured_output": parsed.model_dump(mode="json"), "tools": [],
                "selected_action": "request_clarification" if parsed.missing_fields or parsed.contradictions else "submit_for_safety",
            }
        else:
            try:
                parsed, trace = self.intake_agent.extract_with_trace(
                    state["original_message"], donation_id=state["donation_id"]
                )
            except (LLMError, AgentProtocolError) as exc:
                category = getattr(exc, "category", "unknown")
                provider = getattr(getattr(self.intake_agent, "llm", None), "provider", "gemini")
                model = getattr(getattr(self.intake_agent, "llm", None), "config", None) and getattr(self.intake_agent.llm.config, "llm_model", CANONICAL_MODEL) or CANONICAL_MODEL

                self.database.update_donation(
                    state["donation_id"], donation["status"], DonationStatus.ERROR,
                    "INTAKE_FAILED", "intake_agent",
                    f"Intake extraction failed: {type(exc).__name__}",
                    {"error_type": type(exc).__name__, "category": category},
                    error_message=str(exc),
                )
                self.database.record_agent_execution(
                    state["donation_id"],
                    agent_type="intake_agent",
                    workflow_stage="intake",
                    objective="Extract explicit donor facts and identify only blocking clarification needs",
                    execution_mode="failed",
                    provider=provider,
                    model=model,
                    input_summary={"message_length": len(state["original_message"])},
                    structured_output={},
                    tools=[],
                    selected_action="durable_pause",
                    resulting_status="ERROR",
                    error={"type": type(exc).__name__, "message": str(exc), "category": category},
                    duration_ms=0,
                )
                raise

        needs_clarification = bool(parsed.missing_fields or parsed.contradictions)
        new_status = DonationStatus.NEEDS_CLARIFICATION if needs_clarification else DonationStatus.PENDING_SAFETY_REVIEW
        updated = self.database.update_donation(
            state["donation_id"], donation["status"], new_status,
            "INTAKE_CLARIFICATION_REQUIRED" if needs_clarification else "INTAKE_PARSED",
            "intake_agent",
            "Critical information is missing or contradictory" if needs_clarification else "Structured extraction validated",
            {"missing_fields": parsed.missing_fields, "contradictions": parsed.contradictions, "source": parsed.extraction_source},
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
            execution_mode="deterministic",
            provider="rule_engine",
            model="demo-safety-v1",
            input_summary={"parsed_fields": sorted(state["parsed"].keys())},
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
        try:
            advice = self.receiver_advisor.recommend(donation, ExtractedOffer.model_validate(state["parsed"]))
        except (LLMError, AgentProtocolError) as exc:
            category = getattr(exc, "category", "unknown")
            provider = getattr(getattr(self.receiver_advisor.runtime, "client", None), "provider", "gemini")
            model = getattr(getattr(self.receiver_advisor.runtime, "client", None), "config", None) and getattr(self.receiver_advisor.runtime.client.config, "llm_model", CANONICAL_MODEL) or CANONICAL_MODEL

            self.database.update_donation(
                state["donation_id"], donation["status"], DonationStatus.ERROR,
                "MATCHING_FAILED", "match_route_agent",
                f"Receiver recommendation failed: {type(exc).__name__}",
                {"error_type": type(exc).__name__, "category": category},
                error_message=str(exc),
            )
            self.database.record_agent_execution(
                state["donation_id"],
                agent_type="receiver_advisor_agent",
                workflow_stage="matching",
                objective="Use live receiver tools and recommend only an authoritatively eligible receiver",
                execution_mode="failed",
                provider=provider,
                model=model,
                input_summary={"food_category": state["parsed"].get("food_category"), "attempted_ngo_ids": []},
                structured_output={},
                tools=[],
                selected_action="durable_pause",
                resulting_status="ERROR",
                error={"type": type(exc).__name__, "message": str(exc), "category": category},
                duration_ms=0,
            )
            raise

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

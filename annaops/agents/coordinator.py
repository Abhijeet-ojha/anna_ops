from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from annaops.agents.runtime import AgentProtocolError, BoundedToolRuntime, NoArguments, ToolSpec
from annaops.config import CANONICAL_MODEL, Settings, settings
from annaops.db import Database
from annaops.llm import LLMConfigurationError, LLMError, StructuredLLMClient
from annaops.schemas import AgentDecision, DonationStatus


# Permitted operations the coordinator may propose
PERMITTED_OPERATIONS = {"run_intake", "run_safety", "run_matching", "durable_pause"}

# States that require human input - coordinator must pause
HUMAN_STATUSES = {
    DonationStatus.NEEDS_CLARIFICATION.value,
    DonationStatus.REQUIRES_HUMAN_REVIEW.value,
    DonationStatus.AWAITING_DONOR_APPROVAL.value,
    DonationStatus.AWAITING_NGO_RESPONSE.value,
    DonationStatus.ACCEPTED.value,
}

# Terminal states - no further operations
TERMINAL_STATUSES = {
    DonationStatus.BLOCKED.value,
    DonationStatus.DONOR_REJECTED.value,
    DonationStatus.ESCALATED.value,
    DonationStatus.COMPLETED.value,
    DonationStatus.CANCELLED.value,
    DonationStatus.ERROR.value,
}

# Authoritative status-to-operation mapping (backend enforces this)
OPERATION_BY_STATUS = {
    DonationStatus.INTAKE_PROCESSING.value: "run_intake",
    DonationStatus.PENDING_SAFETY_REVIEW.value: "run_safety",
    DonationStatus.MATCHING.value: "run_matching",
}


class WorkflowCoordinatorAgent:
    """Uses Gemini 3.6 Flash to propose next workflow operations within strict backend gates.

    The model inspects persisted state and proposes an action from a bounded set.
    Backend code validates that the proposed action is permitted for the current status.
    If the model proposes an invalid operation or LLM fails, the coordinator fails explicitly.
    """

    def __init__(self, database: Database, config: Settings = settings, client: StructuredLLMClient | Any | None = None):
        self.database = database
        self.config = config
        self.runtime = BoundedToolRuntime(config, client=client)

    def coordinate(self, donation_id: str) -> str:
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        donation = self.database.get_donation(donation_id)
        status = donation["status"]

        if status in TERMINAL_STATUSES:
            return "durable_pause"

        if not getattr(self.runtime.client, "configured", False):
            raise LLMConfigurationError(
                "Workflow coordinator requires a configured Gemini LLM provider. "
                f"Set ANNAOPS_GEMINI_API_KEY with canonical model '{CANONICAL_MODEL}'."
            )

        return self._coordinate_with_llm(donation, status, started, started_at)

    def _coordinate_with_llm(
        self,
        donation: dict[str, Any],
        status: str,
        started: float,
        started_at: str,
    ) -> str:
        donation_id = donation["id"]

        def read_workflow_state(_: NoArguments) -> dict[str, Any]:
            current = self.database.get_donation(donation_id)
            return {
                "status": current["status"],
                "version": current["version"],
                "has_parsed": current.get("parsed") is not None,
                "has_safety": current.get("safety") is not None,
                "has_match": current.get("match") is not None,
                "selected_ngo_id": current.get("selected_ngo_id"),
                "attempt_count": current.get("attempt_count", 0),
                "donor_approved": current.get("donor_approved", False),
            }

        tools = [
            ToolSpec(
                "read_persisted_workflow_state",
                "Read the current persisted donation status, parsed/safety/match presence, and approval state.",
                read_workflow_state,
            ),
        ]

        try:
            outcome = self.runtime.run(
                agent_name="workflow coordinator",
                objective=(
                    "Inspect the persisted workflow state and select the next permitted operation. "
                    "You must call read_persisted_workflow_state first. "
                    "Permitted operations: run_intake (when status is INTAKE_PROCESSING), "
                    "run_safety (when status is PENDING_SAFETY_REVIEW), "
                    "run_matching (when status is MATCHING), "
                    "durable_pause (when status requires human input or is terminal). "
                    "Do not propose operations that skip required stages."
                ),
                context={"donation_id": donation_id, "current_status": status},
                tools=tools,
                allowed_actions=PERMITTED_OPERATIONS,
                required_tools={"read_persisted_workflow_state"},
            )

            proposed = outcome.decision.proposed_action
            authoritative = OPERATION_BY_STATUS.get(status, "durable_pause")
            if proposed != authoritative:
                if status in HUMAN_STATUSES or status in TERMINAL_STATUSES:
                    if proposed != "durable_pause":
                        raise AgentProtocolError(
                            f"Coordinator proposed '{proposed}' for status '{status}' "
                            f"which requires 'durable_pause'"
                        )
                else:
                    raise AgentProtocolError(
                        f"Coordinator proposed '{proposed}' but status '{status}' requires '{authoritative}'"
                    )

            last_invocation = outcome.invocations[-1] if outcome.invocations else None
            provider = getattr(last_invocation, "provider", "gemini")
            model = getattr(last_invocation, "model", CANONICAL_MODEL)

            self.database.record_agent_execution(
                donation_id,
                agent_type="workflow_coordinator",
                workflow_stage="coordination",
                objective="LLM-assisted planning: choose the next permitted operation",
                execution_mode="llm",
                provider=provider,
                model=model,
                input_summary={"status": status, "version": donation.get("version")},
                structured_output=outcome.decision.model_dump(mode="json"),
                tools=outcome.tool_events,
                selected_action=proposed,
                resulting_status=status,
                duration_ms=round((time.perf_counter() - started) * 1000),
                started_at=started_at,
            )
            return proposed

        except (LLMError, AgentProtocolError) as exc:
            error_decision = AgentDecision(
                current_task="Select the next permitted workflow operation",
                observed_facts=[f"Persisted workflow status is {status}"],
                missing_or_ambiguous=[],
                proposed_action="durable_pause",
                action_arguments={"donation_id": donation_id, "error": type(exc).__name__},
                tool_calls=[],
                evidence=[f"Coordinator planning failed: {type(exc).__name__}"],
                expected_outcome="Workflow paused due to coordinator failure",
                human_input_required=True,
                task_complete=True,
            )
            self.database.record_agent_execution(
                donation_id,
                agent_type="workflow_coordinator",
                workflow_stage="coordination",
                objective="LLM-assisted planning failed",
                execution_mode="failed",
                provider=getattr(getattr(self.runtime, "client", None), "provider", "gemini"),
                model=getattr(getattr(self.runtime, "client", None), "config", None) and getattr(self.runtime.client.config, "llm_model", CANONICAL_MODEL) or CANONICAL_MODEL,
                input_summary={"status": status, "version": donation.get("version")},
                structured_output=error_decision.model_dump(mode="json"),
                tools=[],
                selected_action="durable_pause",
                resulting_status=status,
                error={"type": type(exc).__name__, "message": str(exc), "category": getattr(exc, "category", "agent_protocol_failure")},
                duration_ms=round((time.perf_counter() - started) * 1000),
                started_at=started_at,
            )
            raise

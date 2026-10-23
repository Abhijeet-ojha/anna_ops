from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from annaops.db import Database
from annaops.schemas import AgentDecision, DonationStatus


class WorkflowCoordinatorAgent:
    """Selects the next state-permitted operation without duplicating domain logic."""

    def __init__(self, database: Database):
        self.database = database

    def coordinate(self, donation_id: str) -> str:
        started = time.perf_counter()
        started_at = datetime.now(UTC).isoformat()
        donation = self.database.get_donation(donation_id)
        status = donation["status"]
        operation_by_status = {
            DonationStatus.INTAKE_PROCESSING.value: "run_intake",
            DonationStatus.PENDING_SAFETY_REVIEW.value: "run_safety",
            DonationStatus.MATCHING.value: "run_matching",
        }
        operation = operation_by_status.get(status, "durable_pause")
        human_statuses = {
            DonationStatus.NEEDS_CLARIFICATION.value,
            DonationStatus.REQUIRES_HUMAN_REVIEW.value,
            DonationStatus.AWAITING_DONOR_APPROVAL.value,
            DonationStatus.AWAITING_NGO_RESPONSE.value,
            DonationStatus.ACCEPTED.value,
        }
        terminal_statuses = {
            DonationStatus.BLOCKED.value, DonationStatus.DONOR_REJECTED.value,
            DonationStatus.ESCALATED.value, DonationStatus.COMPLETED.value,
            DonationStatus.CANCELLED.value, DonationStatus.ERROR.value,
        }
        decision = AgentDecision(
            current_task="Select the next permitted workflow operation from persisted state",
            observed_facts=[f"Persisted workflow status is {status}"],
            missing_or_ambiguous=[],
            proposed_action=operation,
            action_arguments={"donation_id": donation_id},
            tool_calls=[],
            evidence=["State transition permissions are enforced by backend status checks"],
            expected_outcome=(
                "Wait for explicit human input" if status in human_statuses else
                "Stop because the workflow is terminal" if status in terminal_statuses else
                f"Execute {operation} once"
            ),
            human_input_required=status in human_statuses,
            task_complete=operation == "durable_pause",
        )
        self.database.record_agent_execution(
            donation_id,
            agent_type="workflow_coordinator",
            workflow_stage="coordination",
            objective="Choose the next permitted operation without crossing safety or human gates",
            execution_mode="deterministic",
            input_summary={"status": status, "version": donation["version"]},
            structured_output=decision.model_dump(mode="json"),
            tools=[{
                "sequence": 1,
                "name": "read_persisted_workflow_state",
                "arguments": {"donation_id": donation_id},
                "purpose": "Avoid relying on in-memory conversation state",
                "result": {"status": status, "version": donation["version"]},
            }],
            selected_action=operation,
            resulting_status=status,
            duration_ms=round((time.perf_counter() - started) * 1000),
            started_at=started_at,
        )
        return operation

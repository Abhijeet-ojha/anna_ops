from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from annaops.agents.matching import MatchRouteAgent
from annaops.agents.runtime import AgentProtocolError, BoundedToolRuntime, NoArguments, ToolSpec
from annaops.config import Settings, settings
from annaops.db import Database
from annaops.llm import LLMResponseError, LLMUnavailable
from annaops.schemas import AgentDecision, ExtractedOffer, MatchResult


@dataclass(frozen=True)
class ReceiverAdvice:
    match: MatchResult
    selected_ngo_id: str | None
    trace: dict[str, Any]


class ReceiverAdvisorAgent:
    """Lets an LLM inspect real matching tools while deterministic constraints remain authoritative."""

    def __init__(self, database: Database, config: Settings = settings):
        self.database = database
        self.config = config
        self.matcher = MatchRouteAgent(config)
        self.runtime = BoundedToolRuntime(config)

    def recommend(
        self,
        donation: dict[str, Any],
        offer: ExtractedOffer,
        excluded_ids: set[str] | None = None,
    ) -> ReceiverAdvice:
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        excluded_ids = excluded_ids or set()
        profiles: list[dict[str, Any]] | None = None
        latest_match: MatchResult | None = None

        def retrieve_profiles(_: NoArguments) -> list[dict[str, Any]]:
            nonlocal profiles
            profiles = self.database.list_ngos()
            return [
                {
                    "id": item["id"], "name": item["name"], "accepted_categories": item["accepted_categories"],
                    "available_capacity": item["available_capacity"], "minimum_quantity": item["minimum_quantity"],
                    "available": item["available"], "opens_at": item["opens_at"], "closes_at": item["closes_at"],
                }
                for item in profiles
            ]

        def inspect_attempts(_: NoArguments) -> dict[str, Any]:
            return {
                "attempt_count": donation["attempt_count"],
                "attempted_ngo_ids": donation["attempted_ngo_ids"],
                "dispatch_attempts": [
                    {
                        "ngo_id": item["ngo_id"], "attempt_number": item["attempt_number"],
                        "status": item["status"], "response_reason": item.get("response_reason"),
                    }
                    for item in donation.get("dispatch_attempts", [])
                ],
            }

        def run_scoring(_: NoArguments) -> dict[str, Any]:
            nonlocal profiles, latest_match
            profiles = profiles or self.database.list_ngos()
            latest_match = self.matcher.match(offer, profiles, excluded_ids)
            return latest_match.model_dump(mode="json")

        tools = [
            ToolSpec("retrieve_receiver_profiles", "Read current receiver capacity, availability, categories, and hours.", retrieve_profiles),
            ToolSpec("inspect_previous_attempts", "Read prior simulated dispatch attempts and responses for this donation.", inspect_attempts),
            ToolSpec("run_eligibility_and_scoring", "Run authoritative eligibility, route estimation, exclusions, and scoring.", run_scoring),
        ]
        fallback_error: dict[str, Any] | None = None
        if self.runtime.client.configured:
            try:
                outcome = self.runtime.run(
                    agent_name="receiver advisor agent",
                    objective=(
                        "Inspect live receiver evidence and recommend one eligible receiver, or escalate only when "
                        "the authoritative tool finds no eligible receiver. Previously attempted receivers cannot be reused."
                    ),
                    context={
                        "donation_status": donation["status"],
                        "offer": offer.model_dump(mode="json"),
                        "attempted_ngo_ids": sorted(excluded_ids),
                        "is_replanning": bool(excluded_ids),
                    },
                    tools=tools,
                    allowed_actions={"recommend_receiver", "escalate"},
                    required_tools={"run_eligibility_and_scoring"},
                )
                if latest_match is None:
                    raise AgentProtocolError("Authoritative matching result was not produced")
                selected = self._validate_final_decision(outcome.decision, latest_match)
                return ReceiverAdvice(
                    match=latest_match,
                    selected_ngo_id=selected,
                    trace={
                        "mode": "llm",
                        "provider": outcome.invocations[-1].provider,
                        "model": outcome.invocations[-1].model,
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                        "retry_count": len(outcome.invocations) - 1,
                        "error": None,
                        "structured_output": outcome.decision.model_dump(mode="json"),
                        "tools": outcome.tool_events,
                        "selected_action": outcome.decision.proposed_action,
                        "started_at": started_at,
                    },
                )
            except (AgentProtocolError, LLMUnavailable, LLMResponseError, ValueError) as exc:
                fallback_error = {"type": type(exc).__name__, "message": str(exc)}

        if latest_match is None:
            retrieve_result = retrieve_profiles(NoArguments())
            score_result = run_scoring(NoArguments())
            tool_events = [
                {"sequence": 1, "name": "retrieve_receiver_profiles", "arguments": {},
                 "purpose": "Load current receiver constraints", "result": retrieve_result},
                {"sequence": 2, "name": "run_eligibility_and_scoring", "arguments": {},
                 "purpose": "Apply authoritative eligibility and scoring", "result": score_result},
            ]
        else:
            tool_events = [{"sequence": 1, "name": "run_eligibility_and_scoring", "arguments": {},
                            "purpose": "Reuse the authoritative result produced before model fallback",
                            "result": latest_match.model_dump(mode="json")}]
        selected = latest_match.candidates[0].ngo_id if latest_match.candidates else None
        decision = AgentDecision(
            current_task="Recommend an eligible receiver from authoritative matching output",
            observed_facts=[f"{len(latest_match.candidates)} eligible receiver(s) remain"],
            missing_or_ambiguous=[],
            proposed_action="recommend_receiver" if selected else "escalate",
            action_arguments={"selected_ngo_id": selected} if selected else {},
            tool_calls=[],
            evidence=[
                f"Authoritative matcher excluded {len(latest_match.exclusions)} receiver(s)",
                "Previously attempted receivers were supplied as hard exclusions",
            ],
            expected_outcome="Fresh donor approval is required" if selected else "Coordinator review is required",
            human_input_required=bool(selected),
            task_complete=True,
        )
        return ReceiverAdvice(
            match=latest_match,
            selected_ngo_id=selected,
            trace={
                "mode": "deterministic_fallback" if fallback_error else "deterministic",
                "provider": self.runtime.client.provider if fallback_error else None,
                "model": self.config.llm_model if fallback_error else None,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "retry_count": 0,
                "error": fallback_error,
                "structured_output": decision.model_dump(mode="json"),
                "tools": tool_events,
                "selected_action": decision.proposed_action,
                "started_at": started_at,
            },
        )

    @staticmethod
    def _validate_final_decision(decision: AgentDecision, match: MatchResult) -> str | None:
        eligible = {candidate.ngo_id for candidate in match.candidates}
        if not eligible:
            if decision.proposed_action != "escalate":
                raise AgentProtocolError("No eligible receiver exists; only escalation is permitted")
            return None
        if decision.proposed_action != "recommend_receiver":
            raise AgentProtocolError("Escalation is not permitted while eligible receivers remain")
        selected = decision.action_arguments.get("selected_ngo_id")
        if selected not in eligible:
            raise AgentProtocolError("Model selected a receiver outside the authoritative eligible set")
        return str(selected)

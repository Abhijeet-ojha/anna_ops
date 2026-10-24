from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from annaops.agents.matching import MatchRouteAgent
from annaops.agents.runtime import AgentProtocolError, BoundedToolRuntime, NoArguments, ToolSpec
from annaops.config import CANONICAL_MODEL, Settings, settings
from annaops.db import Database
from annaops.llm import LLMConfigurationError, LLMError, StructuredLLMClient
from annaops.schemas import AgentDecision, ExtractedOffer, MatchResult


@dataclass(frozen=True)
class ReceiverAdvice:
    match: MatchResult
    selected_ngo_id: str | None
    trace: dict[str, Any]


class ReceiverAdvisorAgent:
    """Uses Gemini 3.6 Flash to inspect live receiver tools while deterministic constraints remain authoritative.

    LLM failures are propagated explicitly; no silent fallback to the highest-ranked candidate.
    """

    def __init__(self, database: Database, config: Settings = settings, client: StructuredLLMClient | Any | None = None):
        self.database = database
        self.config = config
        self.matcher = MatchRouteAgent(config)
        self.runtime = BoundedToolRuntime(config, client=client)

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
                "attempt_count": donation.get("attempt_count", 0),
                "attempted_ngo_ids": donation.get("attempted_ngo_ids", []),
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

        if not getattr(self.runtime.client, "configured", False):
            raise LLMConfigurationError(
                "Receiver advisor agent requires a configured Gemini LLM provider. "
                f"Set ANNAOPS_GEMINI_API_KEY with canonical model '{CANONICAL_MODEL}'."
            )

        outcome = self.runtime.run(
            agent_name="receiver advisor agent",
            objective=(
                "Inspect live receiver evidence and recommend one eligible receiver, or escalate only when "
                "the authoritative tool finds no eligible receiver. Previously attempted receivers cannot be reused."
            ),
            context={
                "donation_status": donation.get("status"),
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

        last_invocation = outcome.invocations[-1] if outcome.invocations else None
        provider = getattr(last_invocation, "provider", "gemini")
        model = getattr(last_invocation, "model", CANONICAL_MODEL)

        return ReceiverAdvice(
            match=latest_match,
            selected_ngo_id=selected,
            trace={
                "mode": "llm",
                "provider": provider,
                "model": model,
                "duration_ms": round((time.perf_counter() - started) * 1000),
                "retry_count": len(outcome.invocations) - 1,
                "error": None,
                "structured_output": outcome.decision.model_dump(mode="json"),
                "tools": outcome.tool_events,
                "selected_action": outcome.decision.proposed_action,
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
        if not selected or selected not in eligible:
            raise AgentProtocolError(
                f"Model selected receiver '{selected}' which is outside the authoritative eligible set: {sorted(eligible)}"
            )
        return str(selected)

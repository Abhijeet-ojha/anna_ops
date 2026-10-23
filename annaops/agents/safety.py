from __future__ import annotations

from datetime import UTC, datetime

from annaops.config import Settings, settings
from annaops.schemas import ExtractedOffer, SafetyAssessment, SafetyVerdict


class SafetyAgent:
    """Deterministic demo rules. No LLM can override this verdict."""

    RECOGNIZED_STORAGE = ("kept hot", "refrigerated", "chilled", "frozen")

    def __init__(self, config: Settings = settings):
        self.config = config

    def assess(self, offer: ExtractedOffer, now: datetime | None = None) -> SafetyAssessment:
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        reasons: list[str] = []

        if offer.contradictions:
            return self._result(SafetyVerdict.BLOCKED, list(offer.contradictions), now)

        critical_missing = [field for field in offer.missing_fields if field != "storage_conditions"]
        if critical_missing:
            reasons.append(f"Critical facts are missing: {', '.join(critical_missing)}.")
            return self._result(SafetyVerdict.NEEDS_CLARIFICATION, reasons, now)

        if not offer.pickup_deadline or offer.pickup_deadline <= now:
            return self._result(SafetyVerdict.BLOCKED, ["The stated pickup deadline has passed."], now)

        if not offer.prepared_at:
            return self._result(SafetyVerdict.NEEDS_CLARIFICATION, ["Preparation time is required."], now)
        if offer.prepared_at > now:
            return self._result(SafetyVerdict.BLOCKED, ["Preparation time is in the future and is contradictory."], now)

        item_units = {item.unit for item in offer.food_items}
        if len(offer.food_items) > 1 and len(item_units) > 1:
            return self._result(
                SafetyVerdict.REQUIRES_HUMAN_REVIEW,
                [
                    "The offer contains multiple food items measured in different units; receiver capacity cannot be compared automatically.",
                    "A coordinator must review item-level quantities before matching.",
                ],
                now,
            )

        if not offer.storage_conditions:
            return self._result(
                SafetyVerdict.REQUIRES_HUMAN_REVIEW,
                ["Storage conditions were not supplied; demo rules cannot automatically establish eligibility."],
                now,
            )
        storage = offer.storage_conditions.lower()
        if "room temperature" in storage:
            return self._result(
                SafetyVerdict.REQUIRES_HUMAN_REVIEW,
                ["Room-temperature storage requires qualified human review; no automatic threshold is assumed."],
                now,
            )
        if not any(marker in storage for marker in self.RECOGNIZED_STORAGE):
            return self._result(
                SafetyVerdict.REQUIRES_HUMAN_REVIEW,
                ["The stated storage method is not recognized by the configured demo rules."],
                now,
            )

        reasons.extend([
            "All configured critical fields are present and internally consistent.",
            "Pickup deadline remains open.",
            f"Storage was explicitly reported as '{offer.storage_conditions}'.",
            "This pass is operational demo eligibility only, not food-safety certification.",
        ])
        return self._result(SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING, reasons, now)

    def _result(self, verdict: SafetyVerdict, reasons: list[str], now: datetime) -> SafetyAssessment:
        return SafetyAssessment(
            verdict=verdict,
            reasons=reasons,
            rule_version=self.config.safety_rule_version,
            assessed_at=now,
        )


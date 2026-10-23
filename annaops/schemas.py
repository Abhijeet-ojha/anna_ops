from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DonationStatus(StrEnum):
    INTAKE_PROCESSING = "INTAKE_PROCESSING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    PENDING_SAFETY_REVIEW = "PENDING_SAFETY_REVIEW"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    BLOCKED = "BLOCKED"
    MATCHING = "MATCHING"
    AWAITING_DONOR_APPROVAL = "AWAITING_DONOR_APPROVAL"
    DONOR_REJECTED = "DONOR_REJECTED"
    APPROVED = "APPROVED"
    AWAITING_NGO_RESPONSE = "AWAITING_NGO_RESPONSE"
    REPLANNING = "REPLANNING"
    ACCEPTED = "ACCEPTED"
    ESCALATED = "ESCALATED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class SafetyVerdict(StrEnum):
    SAFE_TO_PROCEED_TO_MATCHING = "SAFE_TO_PROCEED_TO_MATCHING"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    BLOCKED = "BLOCKED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"


class FoodItem(BaseModel):
    name: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit: str = Field(min_length=1)
    category: Literal["vegetarian", "non_vegetarian", "mixed", "unknown"] = "unknown"


class ExtractedOffer(BaseModel):
    food_type: str | None = None
    food_items: list[FoodItem] = Field(default_factory=list)
    food_category: Literal["vegetarian", "non_vegetarian", "mixed", "unknown"] = "unknown"
    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = None
    prepared_at: datetime | None = None
    pickup_deadline: datetime | None = None
    storage_conditions: str | None = None
    donor_location: str | None = None
    donor_latitude: float | None = Field(default=None, ge=-90, le=90)
    donor_longitude: float | None = Field(default=None, ge=-180, le=180)
    notes: str | None = None
    explicit_fields: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    extraction_source: Literal["configured_llm", "deterministic_demo", "llm_error"] = "deterministic_demo"

    @model_validator(mode="after")
    def check_times(self) -> "ExtractedOffer":
        if self.prepared_at and self.pickup_deadline and self.pickup_deadline <= self.prepared_at:
            message = "Pickup deadline must be after preparation time."
            if message not in self.contradictions:
                self.contradictions.append(message)
        return self


class SafetyAssessment(BaseModel):
    verdict: SafetyVerdict
    reasons: list[str]
    rule_version: str
    assessed_at: datetime
    disclaimer: str = (
        "Demo rules only; this is not regulatory food-safety certification. "
        "A qualified human must validate production criteria."
    )


class RouteEstimate(BaseModel):
    distance_km: float | None
    duration_minutes: float | None
    method: Literal["osrm", "straight_line_fallback", "unavailable"]
    verified_road_route: bool = False
    note: str


class MatchCandidate(BaseModel):
    ngo_id: str
    ngo_name: str
    score: float
    rank: int = 0
    available_capacity: float
    reasons: list[str]
    route: RouteEstimate


class MatchResult(BaseModel):
    candidates: list[MatchCandidate]
    exclusions: dict[str, list[str]]
    rule_version: str
    generated_at: datetime


class DonationCreate(BaseModel):
    message: str = Field(min_length=5, max_length=5000)
    idempotency_key: str | None = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        return value.strip()


class DemoScenarioRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=100)


class ClarificationRequest(BaseModel):
    fields: dict[str, Any]
    idempotency_key: str | None = Field(default=None, max_length=100)

    @field_validator("fields")
    @classmethod
    def require_answers(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("At least one clarification field is required")
        empty = [key for key, answer in value.items() if answer is None or (isinstance(answer, str) and not answer.strip())]
        if empty:
            raise ValueError(f"Clarification values cannot be empty: {', '.join(sorted(empty))}")
        return value


class ActionRequest(BaseModel):
    actor: str = Field(default="demo_donor", min_length=2, max_length=100)
    reason: str | None = Field(default=None, max_length=1000)
    idempotency_key: str | None = Field(default=None, max_length=100)


class NGOResponseRequest(ActionRequest):
    response: Literal["accept", "decline", "timeout"]


class DeliveryConfirmationRequest(ActionRequest):
    verification_note: str = Field(min_length=3, max_length=1000)


class AgentToolCall(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(min_length=1, max_length=500)


class AgentDecision(BaseModel):
    """Validated concise decision record; never stores hidden reasoning or chain-of-thought."""

    current_task: str = Field(min_length=1, max_length=500)
    observed_facts: list[str] = Field(default_factory=list, max_length=30)
    missing_or_ambiguous: list[str] = Field(default_factory=list, max_length=20)
    proposed_action: str = Field(min_length=1, max_length=100)
    action_arguments: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[AgentToolCall] = Field(default_factory=list, max_length=8)
    evidence: list[str] = Field(default_factory=list, max_length=30)
    expected_outcome: str = Field(min_length=1, max_length=1000)
    human_input_required: bool = False
    task_complete: bool = False


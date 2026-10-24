from __future__ import annotations

import urllib.error
from datetime import UTC, datetime, timedelta

import pytest

from annaops.agents.intake import IntakeAgent
from annaops.agents.matching import FICTIONAL_NGOS, MatchRouteAgent
from annaops.agents.safety import SafetyAgent
from annaops.config import Settings
from annaops.schemas import ExtractedOffer, SafetyVerdict
from tests.conftest import complete_message


def test_structured_extraction_tracks_explicit_fields(test_settings: Settings):
    result = IntakeAgent(test_settings).extract(complete_message(test_settings))
    assert result.food_type == "vegetarian biryani"
    assert result.quantity == 40
    assert result.unit == "portions"
    assert result.donor_location == "Demo Campus Jaipur"
    assert result.extraction_source == "configured_llm"
    assert result.missing_fields == []
    assert "prepared_at" in result.explicit_fields


def test_missing_fields_request_targeted_clarification(test_settings: Settings):
    result = IntakeAgent(test_settings).extract("We have leftover food after our event")
    assert "quantity" in result.missing_fields
    assert "prepared_at" in result.missing_fields
    assert any("How much" in question for question in result.clarification_questions)


def test_contradictory_times_are_flagged(test_settings: Settings):
    now = datetime.now(test_settings.timezone)
    result = IntakeAgent(test_settings).extract(
        f"20 portions veg rice prepared at {(now + timedelta(hours=2)).isoformat()}, kept hot, "
        f"pickup by {(now + timedelta(hours=1)).isoformat()} from Demo Hall"
    )
    assert result.contradictions


def test_safety_outcomes_are_deterministic(test_settings: Settings):
    now = datetime.now(UTC)
    base = dict(
        food_type="rice", food_category="vegetarian", quantity=20, unit="portions",
        prepared_at=now - timedelta(minutes=20), pickup_deadline=now + timedelta(hours=2),
        donor_location="Demo Hall", explicit_fields=[], missing_fields=[],
    )
    safety = SafetyAgent(test_settings)
    safe = safety.assess(ExtractedOffer(**base, storage_conditions="kept hot"), now)
    review = safety.assess(ExtractedOffer(**{**base, "missing_fields": ["storage_conditions"]}, storage_conditions=None), now)
    blocked = safety.assess(ExtractedOffer(**{**base, "pickup_deadline": now - timedelta(minutes=1)}, storage_conditions="kept hot"), now)
    assert safe.verdict == SafetyVerdict.SAFE_TO_PROCEED_TO_MATCHING
    assert review.verdict == SafetyVerdict.REQUIRES_HUMAN_REVIEW
    assert blocked.verdict == SafetyVerdict.BLOCKED
    assert safe.rule_version == "demo-safety-v1"


def test_matching_excludes_capacity_and_food_incompatibility(test_settings: Settings):
    now = datetime.now(test_settings.timezone)
    offer = ExtractedOffer(
        food_type="chicken rice", food_category="non_vegetarian", quantity=100, unit="portions",
        prepared_at=now - timedelta(minutes=10), pickup_deadline=now + timedelta(hours=2),
        storage_conditions="kept hot", donor_location="Demo", donor_latitude=26.91,
        donor_longitude=75.78, missing_fields=[],
    )
    result = MatchRouteAgent(test_settings).match(offer, FICTIONAL_NGOS, now=now)
    assert result.candidates
    assert "ngo-03" in result.exclusions
    assert any("not accepted" in reason for reason in result.exclusions["ngo-03"])
    assert "ngo-05" in result.exclusions
    assert any("capacity" in reason for reason in result.exclusions["ngo-05"])
    assert result.candidates[0].route.method == "straight_line_fallback"
    assert result.candidates[0].route.duration_minutes is None


def test_missing_coordinates_do_not_fabricate_route(test_settings: Settings):
    now = datetime.now(test_settings.timezone)
    offer = ExtractedOffer(
        food_type="rice", food_category="vegetarian", quantity=20, unit="portions",
        prepared_at=now - timedelta(minutes=10), pickup_deadline=now + timedelta(hours=2),
        storage_conditions="kept hot", donor_location="Demo", missing_fields=[],
    )
    result = MatchRouteAgent(test_settings).match(offer, FICTIONAL_NGOS, now=now)
    assert result.candidates[0].route.method == "unavailable"
    assert result.candidates[0].route.distance_km is None


def test_configured_llm_failure_raises_explicit_error(monkeypatch, test_settings: Settings):
    from annaops.config import CANONICAL_MODEL
    from annaops.llm import LLMRetryableError
    config = Settings(**{**test_settings.__dict__, "llm_api_key": "test", "llm_model": CANONICAL_MODEL, "testing_mode": False})
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(LLMRetryableError):
        IntakeAgent(config).extract(complete_message(config))


def test_multiple_food_quantities_keep_their_associations(test_settings: Settings):
    result = IntakeAgent(test_settings).extract("65 kgs of chicken biryani and 15 litres of raita")
    assert [(item.name, item.quantity, item.unit) for item in result.food_items] == [
        ("chicken biryani", 65, "kg"),
        ("raita", 15, "litres"),
    ]
    assert result.food_type == "chicken biryani, raita"
    assert result.quantity is None
    assert result.unit is None
    assert "quantity" not in result.missing_fields


def test_legacy_single_quantity_record_is_upgraded_from_original_message(test_settings: Settings):
    agent = IntakeAgent(test_settings)
    legacy = ExtractedOffer(
        food_type="biryani",
        food_category="non_vegetarian",
        quantity=15,
        unit="litres",
        missing_fields=["prepared_at", "pickup_deadline", "donor_location", "storage_conditions"],
        explicit_fields=["food_type", "food_category", "quantity", "unit"],
    )
    upgraded = agent.upgrade_legacy_food_items(
        legacy, "65 kgs of chicken biryani and 15 litres of raita"
    )
    assert [(item.name, item.quantity, item.unit) for item in upgraded.food_items] == [
        ("chicken biryani", 65, "kg"),
        ("raita", 15, "litres"),
    ]
    assert upgraded.quantity is None
    assert upgraded.unit is None
    assert upgraded.missing_fields == legacy.missing_fields


def test_informal_clarification_times_use_configured_local_submission_date(test_settings: Settings):
    agent = IntakeAgent(test_settings)
    current = agent.extract("65 kgs of chicken biryani and 15 litres of raita")
    reference = datetime(2026, 10, 23, 18, 22, tzinfo=UTC)
    merged = agent.merge_clarification(
        current,
        {
            "prepared_at": "6 pm",
            "pickup_deadline": "8 pm",
            "donor_location": "bengaluru",
            "storage_conditions": "cold",
        },
        reference_at=reference,
    )
    assert merged.prepared_at.isoformat() == "2026-10-23T18:00:00+05:30"
    assert merged.pickup_deadline.isoformat() == "2026-10-23T20:00:00+05:30"
    assert merged.donor_location == "bengaluru"
    assert merged.storage_conditions == "cold"
    assert "prepared_at.date_from_submission" in merged.inferred_fields
    assert merged.food_items == current.food_items


@pytest.mark.parametrize("answer", ["6", "evening", "25 pm"])
def test_ambiguous_or_invalid_clarification_time_is_rejected(test_settings: Settings, answer: str):
    agent = IntakeAgent(test_settings)
    current = agent.extract("20 portions vegetarian rice")
    with pytest.raises(ValueError, match="prepared_at"):
        agent.merge_clarification(
            current,
            {"prepared_at": answer},
            reference_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
        )


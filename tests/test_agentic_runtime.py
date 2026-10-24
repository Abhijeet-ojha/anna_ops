from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from annaops.agents.intake import IntakeAgent
from annaops.agents.matching import FICTIONAL_NGOS, MatchRouteAgent
from annaops.agents.receiver_advisor import ReceiverAdvisorAgent
from annaops.config import CANONICAL_MODEL, Settings
from annaops.db import Database
from annaops.llm import LLMInvocation, LLMResponseError
from annaops.main import create_app
from annaops.schemas import AgentDecision, ExtractedOffer
from annaops.agents.runtime import AgentProtocolError
import pytest
from tests.conftest import complete_message


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class SequenceClient:
    configured = True
    provider = "mock.provider"

    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def complete_structured(self, output_model, *, system_prompt, user_payload):
        self.calls.append({"system_prompt": system_prompt, "user_payload": user_payload})
        output = output_model.model_validate(self.outputs.pop(0))
        return LLMInvocation(output=output, provider=self.provider, model=CANONICAL_MODEL, duration_ms=3)


def llm_settings(test_settings: Settings) -> Settings:
    return Settings(**{
        **test_settings.__dict__,
        "llm_api_key": "secret-test-key",
        "llm_base_url": "https://mock.provider/v1",
        "llm_model": CANONICAL_MODEL,
        "testing_mode": False,
    })


def complete_offer(config: Settings) -> ExtractedOffer:
    now = datetime.now(config.timezone)
    return ExtractedOffer(
        food_type="vegetarian biryani",
        food_items=[{"name": "vegetarian biryani", "quantity": 40, "unit": "portions", "category": "vegetarian"}],
        food_category="vegetarian", quantity=40, unit="portions",
        prepared_at=now - timedelta(minutes=20), pickup_deadline=now + timedelta(hours=3),
        storage_conditions="kept hot", donor_location="Demo Campus Jaipur",
        donor_latitude=26.9124, donor_longitude=75.7873,
        explicit_fields=["food_items", "food_type", "food_category", "quantity", "unit", "prepared_at",
                         "pickup_deadline", "storage_conditions", "donor_location", "donor_latitude", "donor_longitude"],
    )


def decision(action: str, *, arguments: dict | None = None, tools: list[dict] | None = None) -> dict:
    return AgentDecision(
        current_task="Inspect current receiver evidence",
        observed_facts=["Use authoritative backend results"],
        proposed_action=action,
        action_arguments=arguments or {},
        tool_calls=tools or [],
        evidence=["Decision is constrained by tool output"],
        expected_outcome="Produce a validated next step",
        human_input_required=action == "recommend_receiver",
        task_complete=not tools,
    ).model_dump(mode="json")


def test_configured_intake_really_invokes_provider_and_validates(monkeypatch, test_settings: Settings):
    config = llm_settings(test_settings)
    offer = complete_offer(config)
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({
            "id": "mock-response-1",
            "choices": [{"message": {"content": offer.model_dump_json()}}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result, trace = IntakeAgent(config).extract_with_trace(complete_message(config))

    assert captured["url"] == "https://mock.provider/v1/chat/completions"
    assert captured["body"]["model"] == CANONICAL_MODEL
    assert captured["body"]["response_format"]["type"] == "json_schema"
    assert captured["body"]["response_format"]["json_schema"]["name"] == "extractedoffer"
    assert result.extraction_source == "configured_llm"
    assert result.food_items[0].quantity == 40
    assert trace["mode"] == "llm"
    assert trace["model"] == CANONICAL_MODEL


def test_llm_cannot_hide_required_missing_fields(monkeypatch, test_settings: Settings):
    config = llm_settings(test_settings)
    model_claim = ExtractedOffer(
        food_type="rice",
        food_items=[{"name": "rice", "quantity": 20, "unit": "portions"}],
        quantity=20,
        unit="portions",
        explicit_fields=["food_type", "food_items", "quantity", "unit"],
        missing_fields=[],
        clarification_questions=["Where and when can this be collected?"],
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeHTTPResponse({
            "choices": [{"message": {"content": model_claim.model_dump_json()}}]
        }),
    )

    result = IntakeAgent(config).extract("20 portions rice")

    assert {"prepared_at", "pickup_deadline", "donor_location", "storage_conditions"}.issubset(result.missing_fields)
    assert result.extraction_source == "configured_llm"
    assert result.clarification_questions


def test_malformed_llm_output_raises_explicit_error(monkeypatch, test_settings: Settings):
    config = llm_settings(test_settings)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: FakeHTTPResponse({"choices": [{"message": {"content": "not-json"}}]}),
    )

    with pytest.raises(LLMResponseError):
        IntakeAgent(config).extract_with_trace(complete_message(config))


def test_receiver_agent_uses_tools_then_adapts_to_results(tmp_path: Path, test_settings: Settings):
    config = Settings(**{**llm_settings(test_settings).__dict__, "database_path": tmp_path / "agent-tools.db"})
    database = Database(config)
    database.initialize()
    database.seed_ngos(FICTIONAL_NGOS)
    donation = database.create_donation("synthetic complete offer", "MATCHING")
    offer = complete_offer(config)
    authoritative = MatchRouteAgent(config).match(offer, database.list_ngos())
    selected = authoritative.candidates[1].ngo_id
    client = SequenceClient([
        decision("use_tools", tools=[
            {"name": "retrieve_receiver_profiles", "arguments": {}, "purpose": "Inspect current capacity"},
            {"name": "run_eligibility_and_scoring", "arguments": {}, "purpose": "Obtain eligible candidates"},
        ]),
        decision("recommend_receiver", arguments={"selected_ngo_id": selected}),
    ])
    advisor = ReceiverAdvisorAgent(database, config)
    advisor.runtime.client = client

    advice = advisor.recommend(database.get_donation(donation["id"]), offer)

    assert len(client.calls) == 2
    assert [event["name"] for event in advice.trace["tools"]] == [
        "retrieve_receiver_profiles", "run_eligibility_and_scoring"
    ]
    profile_event, scoring_event = advice.trace["tools"]
    assert profile_event["argument_source"] == "backend_bound"
    assert profile_event["arguments"]["donation_id"] == donation["id"]
    assert scoring_event["argument_source"] == "backend_bound"
    assert scoring_event["arguments"]["offered_food"] == ["40 portions of vegetarian biryani"]
    assert scoring_event["arguments"]["storage_conditions"] == "kept hot"
    assert scoring_event["arguments"]["donor_location"] == "Demo Campus Jaipur"
    assert advice.selected_ngo_id == selected
    assert advice.trace["mode"] == "llm"
    assert client.calls[1]["user_payload"]["tool_results_so_far"]


def test_invalid_model_receiver_action_raises_agent_protocol_error(tmp_path: Path, test_settings: Settings):
    config = Settings(**{**llm_settings(test_settings).__dict__, "database_path": tmp_path / "invalid-action.db"})
    database = Database(config)
    database.initialize()
    database.seed_ngos(FICTIONAL_NGOS)
    donation = database.create_donation("synthetic complete offer", "MATCHING")
    offer = complete_offer(config)
    client = SequenceClient([
        decision("use_tools", tools=[
            {"name": "run_eligibility_and_scoring", "arguments": {}, "purpose": "Check eligibility"},
        ]),
        decision("recommend_receiver", arguments={"selected_ngo_id": "ngo-08"}),
    ])
    advisor = ReceiverAdvisorAgent(database, config, client=client)

    with pytest.raises(AgentProtocolError, match="outside the authoritative eligible set"):
        advisor.recommend(database.get_donation(donation["id"]), offer)


def test_replanning_tools_exclude_attempted_receiver(tmp_path: Path, test_settings: Settings):
    config = Settings(**{**test_settings.__dict__, "database_path": tmp_path / "replan-tools.db"})
    database = Database(config)
    database.initialize()
    database.seed_ngos(FICTIONAL_NGOS)
    offer = complete_offer(config)
    donation = database.create_donation("synthetic complete offer", "MATCHING")
    first = MatchRouteAgent(config).match(offer, database.list_ngos()).candidates[0].ngo_id
    database.update_donation(
        donation["id"], "MATCHING", "MATCHING", "TEST_ATTEMPT", "test", "Set attempted receiver",
        attempted_ngo_ids_json=[first], attempt_count=1,
    )

    advice = ReceiverAdvisorAgent(database, config).recommend(
        database.get_donation(donation["id"]), offer, {first}
    )

    assert advice.selected_ngo_id != first
    assert first in advice.match.exclusions
    assert "Already attempted" in advice.match.exclusions[first][0]


def test_agent_executions_survive_application_restart(tmp_path: Path, test_settings: Settings):
    config = Settings(**{**test_settings.__dict__, "database_path": tmp_path / "persistent-agent.db"})
    first_app = create_app(config)
    with TestClient(first_app) as client:
        created = client.post("/api/donations", json={
            "message": complete_message(config), "idempotency_key": "agent-persistence-key"
        }).json()
        executions = client.get(f"/api/donations/{created['id']}/agent-executions").json()
        assert any(item["agent_type"] == "workflow_coordinator" for item in executions)
        assert any(item["agent_type"] == "receiver_advisor_agent" for item in executions)

    restarted_app = create_app(config)
    with TestClient(restarted_app) as client:
        persisted = client.get(f"/api/donations/{created['id']}/agent-executions").json()
        assert [item["id"] for item in persisted] == [item["id"] for item in executions]


def test_dashboard_explicitly_renders_llm_and_fallback_modes():
    source = Path("static/app.js").read_text(encoding="utf-8")
    assert 'mode === "llm"' in source
    assert "llm-success" in source
    assert "llm-failed" in source
    assert "Actual tool sequence" in source
    assert "Inputs used" in source
    assert "Backend-bound · validated" in source


def test_gemini_environment_is_the_first_class_provider(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ANNAOPS_GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("ANNAOPS_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("ANNAOPS_GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("ANNAOPS_LLM_API_KEY", "legacy-key")
    monkeypatch.setenv("ANNAOPS_LLM_MODEL", "legacy-model")

    config = Settings.from_env()

    assert config.llm_api_key == "gemini-test-key"
    assert config.llm_model == CANONICAL_MODEL
    assert config.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

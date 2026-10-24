from __future__ import annotations

import json
import os
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from annaops.agents.coordinator import WorkflowCoordinatorAgent
from annaops.agents.intake import IntakeAgent
from annaops.agents.matching import FICTIONAL_NGOS
from annaops.agents.receiver_advisor import ReceiverAdvisorAgent
from annaops.agents.runtime import AgentProtocolError
from annaops.config import CANONICAL_BASE_URL, CANONICAL_MODEL, ConfigurationError, Settings
from annaops.db import Database
from annaops.llm import (
    LLMConfigurationError,
    LLMInvocation,
    LLMResponseError,
    LLMRetryableError,
    StructuredLLMClient,
    TestMockClient,
)
from annaops.main import create_app
from annaops.schemas import AgentDecision, ExtractedOffer, FoodItem


class MockResponse:
    def __init__(self, data: dict, status: int = 200):
        self.data = data
        self.status = status

    def read(self) -> bytes:
        return json.dumps(self.data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_startup_fails_when_api_key_is_missing(tmp_path: Path):
    """When ANNAOPS_GEMINI_API_KEY is missing, application validation must fail."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-no-key.db",
        llm_api_key="",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )
    with pytest.raises(ConfigurationError, match="ANNAOPS_GEMINI_API_KEY"):
        config.validate()


def test_startup_fails_when_model_is_not_canonical(tmp_path: Path):
    """Silent substitution of another model like gemini-2.5-flash or gemini-3.8-flash is forbidden."""
    for invalid_model in ["gemini-2.5-flash", "gemini-3.8-flash", "gpt-4o", "claude-3-5-sonnet"]:
        config = Settings.for_testing(
            database_path=tmp_path / "test-wrong-model.db",
            llm_api_key="valid-dummy-key",
            llm_model=invalid_model,
            testing_mode=False,
        )
        with pytest.raises(ConfigurationError, match="gemini-3.6-flash"):
            config.validate()


def test_canonical_model_is_used_in_provider_requests(monkeypatch, tmp_path: Path):
    """Verify that gemini-3.6-flash is strictly used in provider HTTP requests."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-canonical-req.db",
        llm_api_key="valid-dummy-key",
        llm_model=CANONICAL_MODEL,
        llm_base_url=CANONICAL_BASE_URL,
        testing_mode=False,
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        offer = ExtractedOffer(
            food_type="vegetarian biryani",
            food_items=[FoodItem(name="vegetarian biryani", quantity=40, unit="portions", category="vegetarian")],
            food_category="vegetarian",
            quantity=40,
            unit="portions",
            prepared_at=datetime.now(),
            pickup_deadline=datetime.now() + timedelta(hours=3),
            storage_conditions="kept hot",
            donor_location="Demo Campus",
            donor_latitude=26.9124,
            donor_longitude=75.7873,
            explicit_fields=["food_type", "quantity", "unit"],
            missing_fields=[],
            clarification_questions=[],
            extraction_source="configured_llm",
        )
        return MockResponse({
            "id": "chatcmpl-test-canonical",
            "choices": [{"message": {"content": offer.model_dump_json()}}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(config)
    res = client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")

    assert captured["body"]["model"] == "gemini-3.6-flash"
    assert captured["url"] == f"{CANONICAL_BASE_URL}/chat/completions"
    assert res.model == "gemini-3.6-flash"
    assert res.provider == "gemini"


def test_provider_authentication_error_mapped(monkeypatch, tmp_path: Path):
    """401 or invalid key error must be surfaced as an LLMConfigurationError."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-auth.db",
        llm_api_key="invalid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("https://fake", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(config)
    with pytest.raises(LLMConfigurationError, match="Authentication failed"):
        client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")


def test_unsupported_model_error_mapped(monkeypatch, tmp_path: Path):
    """404 Model Not Found must be surfaced as an LLMConfigurationError."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-not-found.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.HTTPError("https://fake", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(config)
    with pytest.raises(LLMConfigurationError, match="not found"):
        client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")


def test_rate_limit_and_timeout_mapped_as_retryable(monkeypatch, tmp_path: Path):
    """429 and timeouts must be surfaced as retryable LLMRetryableError."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-retryable.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_429(*args, **kwargs):
        raise urllib.error.HTTPError("https://fake", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_429)
    client = StructuredLLMClient(config)
    with pytest.raises(LLMRetryableError, match="Rate limit exceeded"):
        client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")

    def fake_timeout(*args, **kwargs):
        raise TimeoutError("Connection timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_timeout)
    with pytest.raises(LLMRetryableError, match="unreachable"):
        client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")


def test_malformed_structured_response_mapped(monkeypatch, tmp_path: Path):
    """Malformed JSON response must be surfaced as an LLMResponseError."""
    config = Settings.for_testing(
        database_path=tmp_path / "test-malformed.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_urlopen(*args, **kwargs):
        return MockResponse({"choices": [{"message": {"content": "not-a-valid-json-string"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = StructuredLLMClient(config)
    with pytest.raises(LLMResponseError, match="invalid structured response"):
        client.complete_structured(ExtractedOffer, system_prompt="Test", user_payload="Test")


def test_invalid_receiver_recommendation_fails_explicitly(tmp_path: Path):
    """Model proposing an ineligible receiver must fail the agent operation with AgentProtocolError."""
    config = Settings.for_testing(database_path=tmp_path / "ineligible-test.db")
    database = Database(config)
    database.initialize()
    database.seed_ngos(FICTIONAL_NGOS)
    donation = database.create_donation("test offer", "MATCHING")

    offer = ExtractedOffer(
        food_type="biryani",
        food_category="vegetarian",
        quantity=30,
        unit="portions",
        prepared_at=datetime.now(),
        pickup_deadline=datetime.now() + timedelta(hours=2),
        storage_conditions="kept hot",
        donor_location="Demo Campus",
        donor_latitude=26.9124,
        donor_longitude=75.7873,
        explicit_fields=[],
        missing_fields=[],
    )

    class IneligibleClient:
        configured = True
        provider = "test"
        model = CANONICAL_MODEL
        def complete_structured(self, *args, **kwargs):
            return LLMInvocation(
                output=AgentDecision(
                    current_task="Recommend ineligible",
                    observed_facts=[],
                    missing_or_ambiguous=[],
                    proposed_action="recommend_receiver",
                    action_arguments={"selected_ngo_id": "ngo-invalid-999"},
                    tool_calls=[],
                    evidence=[],
                    expected_outcome="Ineligible outcome",
                    human_input_required=True,
                    task_complete=True,
                ),
                provider=self.provider,
                model=self.model,
                duration_ms=5,
            )

    advisor = ReceiverAdvisorAgent(database, config, client=IneligibleClient())
    with pytest.raises(AgentProtocolError):
        advisor.recommend(donation, offer)


def test_failed_agent_executions_are_persisted_and_visible(monkeypatch, tmp_path: Path):
    """When an agent operation fails, execution record is persisted and visible via the API."""
    config = Settings.for_testing(
        database_path=tmp_path / "persisted-failure.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_urlopen(*args, **kwargs):
        raise urllib.error.URLError("Provider disconnected")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(config)

    with TestClient(app) as client:
        res = client.post("/api/donations", json={"message": "30 portions veg rice, kept hot"})
        assert res.status_code == 503

        # List donations and verify state
        donations = client.get("/api/donations").json()
        assert len(donations) == 1
        d = donations[0]
        assert d["status"] == "ERROR"

        # Check agent execution record
        executions = client.get(f"/api/donations/{d['id']}/agent-executions").json()
        assert len(executions) >= 1
        failed_exec = executions[0]
        assert failed_exec["model"] == "gemini-3.6-flash"
        assert failed_exec["provider"] == "gemini"
        assert failed_exec["resulting_status"] in ("INTAKE_PROCESSING", "ERROR")
        assert failed_exec["error"]["category"] == "retryable_provider_error" 


def test_workflow_state_remains_safe_after_failure(monkeypatch, tmp_path: Path):
    """A failed operation must never advance to matching or approval."""
    config = Settings.for_testing(
        database_path=tmp_path / "safe-state.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=False,
    )

    def fake_urlopen(*args, **kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = create_app(config)

    with TestClient(app) as client:
        client.post("/api/donations", json={"message": "30 portions veg rice"})
        donations = client.get("/api/donations").json()
        donation = donations[0]
        assert donation["status"] == "ERROR"
        assert donation.get("selected_ngo_id") is None
        assert donation.get("donor_approved") in (False, None)


def test_successful_retry_after_provider_recovery(monkeypatch, tmp_path: Path):
    """When provider recovers, calling POST /api/donations/{id}/retry must successfully resume."""
    config = Settings.for_testing(
        database_path=tmp_path / "retry-success.db",
        llm_api_key="valid-key",
        llm_model=CANONICAL_MODEL,
        testing_mode=True,
    )
    database = Database(config)
    database.initialize()
    database.seed_ngos(FICTIONAL_NGOS)

    # First, simulate a failure that placed donation in ERROR
    now = datetime.now(config.timezone).replace(microsecond=0)
    prepared = (now - timedelta(minutes=20)).isoformat()
    deadline = (now + timedelta(hours=3)).isoformat()
    message = (
        f"40 portions vegetarian biryani prepared at {prepared}, kept hot, "
        f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
    )

    donation, _ = database.create_donation_once(message, "INTAKE_PROCESSING", "retry-test-key", synthetic=True)
    database.update_donation(
        donation["id"], "INTAKE_PROCESSING", "ERROR", "WORKFLOW_ERROR", "system",
        "Failed before retry", {"error": "offline"}, error_message="offline",
    )

    app = create_app(config, llm_client=TestMockClient(config))
    with TestClient(app) as client:
        # Call retry endpoint
        res = client.post(f"/api/donations/{donation['id']}/retry")
        assert res.status_code == 200, res.text
        updated = res.json()
        assert updated["status"] == "AWAITING_DONOR_APPROVAL"
        assert updated["selected_ngo_id"] is not None


@pytest.mark.live_llm
def test_live_gemini_smoke_test():
    """Explicitly opt-in live-provider smoke test using real Gemini 3.6 Flash.

    Run with: pytest -m live_llm
    Requires ANNAOPS_GEMINI_API_KEY environment variable.
    """
    api_key = os.getenv("ANNAOPS_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("ANNAOPS_GEMINI_API_KEY is not set in environment.")

    config = Settings.for_testing(
        llm_api_key=api_key,
        llm_model=CANONICAL_MODEL,
        llm_base_url=CANONICAL_BASE_URL,
        testing_mode=False,
    )
    client = StructuredLLMClient(config)

    now = datetime.now()
    message = (
        f"40 portions vegetarian biryani prepared at {(now - timedelta(minutes=20)).isoformat()}, kept hot, "
        f"pickup by {(now + timedelta(hours=2)).isoformat()} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
    )

    invocation = None
    last_err = None
    import time
    for attempt in range(3):
        try:
            invocation = client.complete_structured(
                ExtractedOffer,
                system_prompt="You are the AnnaOps intake agent. Extract structured facts into ExtractedOffer schema.",
                user_payload={"original_donor_message": message},
            )
            break
        except LLMRetryableError as err:
            last_err = err
            time.sleep(2)

    if invocation is None:
        raise last_err

    assert invocation.provider == "gemini"
    assert invocation.model == "gemini-3.6-flash"
    assert invocation.output.food_type is not None
    assert invocation.output.quantity is not None

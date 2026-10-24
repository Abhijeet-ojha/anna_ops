from __future__ import annotations

import json
import threading
import urllib.error
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from annaops.agents.matching import FICTIONAL_NGOS
from annaops.config import CANONICAL_MODEL, Settings
from annaops.db import BudgetExceeded, Database
from annaops.llm import LLMInvocation, LLMRetryableError, StructuredLLMClient, TestMockClient
from annaops.main import create_app
from annaops.schemas import ExtractedOffer
from tests.conftest import complete_message


class Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def provider_settings(tmp_path, **overrides) -> Settings:
    values = {
        **Settings.for_testing(database_path=tmp_path / "quota.db").__dict__,
        "testing_mode": False,
        "llm_api_key": "test-key",
        "llm_model": CANONICAL_MODEL,
        "llm_max_retries": 2,
    }
    values.update(overrides)
    return Settings(**values)


def valid_offer(config: Settings) -> ExtractedOffer:
    now = datetime.now(config.timezone)
    return ExtractedOffer(
        food_type="vegetarian biryani",
        food_items=[{"name": "vegetarian biryani", "quantity": 40, "unit": "portions", "category": "vegetarian"}],
        food_category="vegetarian",
        quantity=40,
        unit="portions",
        prepared_at=now - timedelta(minutes=20),
        pickup_deadline=now + timedelta(hours=3),
        storage_conditions="kept hot",
        donor_location="Demo Campus Jaipur",
        explicit_fields=["food_type", "quantity", "unit"],
        extraction_source="configured_llm",
    )


def test_complete_workflow_uses_measured_five_model_rounds(tmp_path):
    config = Settings.for_testing(database_path=tmp_path / "workflow.db")
    client = TestMockClient(config)
    app = create_app(config, llm_client=client)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/donations",
            json={"message": complete_message(config), "idempotency_key": "measured-call-key"},
        )
    assert response.status_code == 201
    assert len(client.calls) == 5


def test_budget_reservation_is_atomic_under_concurrency(tmp_path):
    config = Settings.for_testing(database_path=tmp_path / "budget.db")
    database = Database(config)
    database.initialize()
    donation = database.create_donation("synthetic offer", "INTAKE_PROCESSING")
    successes: list[str] = []
    failures: list[Exception] = []

    def reserve():
        try:
            successes.append(database.reserve_llm_request(
                donation["id"], "test", "test", 5, "gemini", CANONICAL_MODEL,
            ))
        except Exception as exc:
            failures.append(exc)

    threads = [threading.Thread(target=reserve) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 5
    assert len(failures) == 5
    assert all(isinstance(error, BudgetExceeded) for error in failures)


def test_budget_error_names_the_selected_provider(tmp_path):
    config = Settings(**{
        **Settings.for_testing(database_path=tmp_path / "provider-budget.db").__dict__,
        "llm_provider": "groq",
        "llm_budget_per_donation": 1,
    })
    database = Database(config)
    database.initialize()
    donation = database.create_donation("provider budget test", "INTAKE_PROCESSING")
    database.reserve_llm_request(
        donation["id"], "intake_agent", "intake", 1, "groq", "openai/gpt-oss-20b",
    )
    with pytest.raises(BudgetExceeded, match="Groq request budget exhausted"):
        database.reserve_llm_request(
            donation["id"], "intake_agent", "intake", 1, "groq", "openai/gpt-oss-20b",
        )


def test_429_retries_are_bounded_and_counted(monkeypatch, tmp_path):
    config = provider_settings(tmp_path)
    database = Database(config)
    database.initialize()
    donation = database.create_donation("synthetic offer", "INTAKE_PROCESSING")
    calls = 0

    def rate_limited(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError("https://fake", 429, "Too Many Requests", {"Retry-After": "0"}, None)

    monkeypatch.setattr("urllib.request.urlopen", rate_limited)
    monkeypatch.setattr("annaops.llm.time", type("Clock", (), {
        "perf_counter": staticmethod(__import__("time").perf_counter),
        "sleep": staticmethod(lambda _: None),
    }))

    with pytest.raises(LLMRetryableError):
        StructuredLLMClient(config, database=database).complete_structured(
            ExtractedOffer, system_prompt="test", user_payload="test",
            donation_id=donation["id"], agent_type="test", workflow_stage="test",
        )

    assert calls == 3
    usage = database.llm_usage(donation["id"], config.llm_budget_per_donation)
    assert usage["request_count"] == 3
    assert usage["retry_count"] == 2
    assert usage["rate_limit_errors"] == 3


def test_non_retryable_authentication_error_uses_one_request(monkeypatch, tmp_path):
    config = provider_settings(tmp_path)
    database = Database(config)
    database.initialize()
    donation = database.create_donation("synthetic offer", "INTAKE_PROCESSING")

    def unauthorized(*args, **kwargs):
        raise urllib.error.HTTPError("https://fake", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", unauthorized)
    with pytest.raises(Exception, match="Authentication failed"):
        StructuredLLMClient(config, database=database).complete_structured(
            ExtractedOffer, system_prompt="test", user_payload="test",
            donation_id=donation["id"], agent_type="test", workflow_stage="test",
        )

    usage = database.llm_usage(donation["id"], config.llm_budget_per_donation)
    assert usage["request_count"] == 1
    assert usage["retry_count"] == 0

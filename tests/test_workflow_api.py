from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from annaops.config import Settings
from annaops.db import Database
from annaops.main import create_app
from tests.conftest import complete_message


def create_offer(client: TestClient, config: Settings):
    response = client.post("/api/donations", json={"message": complete_message(config)})
    assert response.status_code == 201, response.text
    return response.json()


def test_successful_end_to_end_requires_separate_delivery_confirmation(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    assert offer["status"] == "AWAITING_DONOR_APPROVAL"
    assert offer["safety"]["verdict"] == "SAFE_TO_PROCEED_TO_MATCHING"
    assert offer["match"]["candidates"]

    approved = client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_donor"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "AWAITING_NGO_RESPONSE"

    accepted = client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "accept"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "ACCEPTED"
    assert accepted.json()["delivery_confirmed_at"] is None

    confirmed = client.post(
        f"/api/donations/{offer['id']}/delivery-confirmation",
        json={"actor": "demo_coordinator", "verification_note": "Synthetic handoff checked by demo operator"},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "COMPLETED"
    assert confirmed.json()["final_outcome"]["delivery"] == "confirmed_in_demo"


def test_dispatch_cannot_happen_without_approval(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    response = client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "accept"},
    )
    assert response.status_code == 409
    assert response.json()["error"] == "invalid_transition"


def test_rejected_first_ngo_replans_and_requires_fresh_approval(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    first_ngo = offer["selected_ngo_id"]
    client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_donor"})
    declined = client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "decline"},
    )
    assert declined.status_code == 200
    body = declined.json()
    assert body["status"] == "AWAITING_DONOR_APPROVAL"
    assert body["selected_ngo_id"] != first_ngo
    assert first_ngo in body["attempted_ngo_ids"]
    assert body["donor_approved"] is False


def test_second_timeout_escalates(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_donor"})
    first = client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "timeout"},
    ).json()
    assert first["status"] == "AWAITING_DONOR_APPROVAL"
    client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_donor"})
    second = client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "timeout"},
    ).json()
    assert second["status"] == "ESCALATED"
    assert second["attempt_count"] == 2


def test_create_and_action_idempotency(client: TestClient, test_settings: Settings):
    key = "fixed-create-key"
    message = complete_message(test_settings)
    first = client.post("/api/donations", json={"message": message, "idempotency_key": key}).json()
    second = client.post("/api/donations", json={"message": message, "idempotency_key": key}).json()
    assert first["id"] == second["id"]

    approve_key = "fixed-approve-key"
    first_approval = client.post(
        f"/api/donations/{first['id']}/approve", json={"actor": "demo_donor", "idempotency_key": approve_key}
    )
    replay = client.post(
        f"/api/donations/{first['id']}/approve", json={"actor": "demo_donor", "idempotency_key": approve_key}
    )
    assert first_approval.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["attempt_count"] == 1


def test_incomplete_offer_can_be_clarified(client: TestClient, test_settings: Settings):
    incomplete = client.post("/api/donations", json={"message": "We have vegetarian rice left after an event"}).json()
    assert incomplete["status"] == "NEEDS_CLARIFICATION"
    complete = create_offer(client, test_settings)["parsed"]
    fields = {key: complete[key] for key in incomplete["parsed"]["missing_fields"]}
    clarified = client.post(f"/api/donations/{incomplete['id']}/clarify", json={"fields": fields})
    assert clarified.status_code == 200, clarified.text
    assert clarified.json()["status"] == "AWAITING_DONOR_APPROVAL"


def test_unknown_storage_requires_review_and_cannot_be_approved(client: TestClient, test_settings: Settings):
    unknown = client.post(
        "/api/donations",
        json={"message": complete_message(test_settings, storage="room temperature")},
    )
    assert unknown.status_code == 201
    body = unknown.json()
    assert body["status"] == "REQUIRES_HUMAN_REVIEW"
    assert "requires qualified human review" in body["safety"]["reasons"][0]
    approval = client.post(
        f"/api/donations/{body['id']}/safety-review",
        json={"actor": "demo_coordinator", "decision": "approve", "reason": "Review attempted without new evidence"},
    )
    assert approval.status_code == 422
    assert "revalidation remains" in approval.json()["detail"]
    rejected = client.post(
        f"/api/donations/{body['id']}/safety-review",
        json={
            "actor": "demo_coordinator",
            "decision": "reject",
            "reason": "Storage method could not be verified",
            "idempotency_key": "review-reject-1",
        },
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "BLOCKED"
    assert rejected.json()["audit_events"][-1]["event_type"] == "SAFETY_REVIEW_REJECTED"


def test_safety_review_can_request_clarification_then_revalidate(client: TestClient, test_settings: Settings):
    response = client.post(
        "/api/donations",
        json={"message": complete_message(test_settings, storage="room temperature")},
    )
    donation = response.json()
    requested = client.post(
        f"/api/donations/{donation['id']}/safety-review",
        json={
            "actor": "demo_coordinator",
            "decision": "request_clarification",
            "reason": "Please provide the recognized storage method used for this offer",
        },
    )
    assert requested.status_code == 200
    requested_body = requested.json()
    assert requested_body["status"] == "NEEDS_CLARIFICATION"
    assert "storage_conditions" in requested_body["parsed"]["missing_fields"]
    assert requested_body["parsed"]["storage_conditions"] is None
    calls_before_clarification = len(client.app.state.workflow.intake_agent.llm.calls)
    invalid = client.post(
        f"/api/donations/{donation['id']}/clarify",
        json={"fields": {"storage_conditions": "normal"}},
    )
    assert invalid.status_code == 422
    assert "Unsupported storage condition" in invalid.json()["detail"]
    unchanged = client.get(f"/api/donations/{donation['id']}").json()
    assert unchanged["status"] == "NEEDS_CLARIFICATION"
    assert unchanged["parsed"]["storage_conditions"] is None
    clarified = client.post(
        f"/api/donations/{donation['id']}/clarify",
        json={"fields": {"storage_conditions": "refrigerated"}},
    )
    assert clarified.status_code == 200
    assert clarified.json()["status"] == "AWAITING_DONOR_APPROVAL"
    assert len(client.app.state.workflow.intake_agent.llm.calls) - calls_before_clarification == 2


def test_audit_csv_health_and_role_gate(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    audit = client.get(f"/api/donations/{offer['id']}/audit")
    assert audit.status_code == 200
    assert {event["event_type"] for event in audit.json()} >= {"DONATION_CREATED", "SAFETY_ASSESSED", "MATCHING_COMPLETED"}
    record = client.get(f"/api/donations/{offer['id']}/record.csv")
    assert record.status_code == 200
    assert "Synthetic" in record.text or "synthetic" in record.text
    health = client.get("/api/health").json()
    assert health["demo_mode"] is True
    assert health["routing"] == "straight_line_fallback"
    forbidden = client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_ngo"})
    assert forbidden.status_code == 403


def test_sqlite_persists_across_app_restart(test_settings: Settings):
    app_one = create_app(test_settings)
    with TestClient(app_one) as first_client:
        offer = create_offer(first_client, test_settings)
    app_two = create_app(test_settings)
    with TestClient(app_two) as second_client:
        restored = second_client.get(f"/api/donations/{offer['id']}")
        assert restored.status_code == 200
        assert restored.json()["status"] == "AWAITING_DONOR_APPROVAL"
        assert restored.json()["audit_events"]


def test_database_capacity_constraint(test_settings: Settings):
    database = Database(test_settings)
    database.initialize()
    try:
        with database.transaction() as connection:
            connection.execute(
                "INSERT INTO ngo_profiles VALUES ('bad','Bad','Demo',0,0,'[]',1,-1,'00:00','23:59',1,0,1)"
            )
    except Exception as exc:
        assert "CHECK constraint" in str(exc)
    else:
        raise AssertionError("Negative capacity was accepted")


def test_llm_failure_stops_before_safety_and_matching(monkeypatch, test_settings: Settings):
    import urllib.error
    from annaops.config import CANONICAL_MODEL

    configured = Settings(**{**test_settings.__dict__, "llm_api_key": "test", "llm_model": CANONICAL_MODEL, "testing_mode": False})
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    app = create_app(configured)
    with TestClient(app) as configured_client:
        response = configured_client.post("/api/donations", json={"message": complete_message(configured)})
        assert response.status_code == 503
        body = response.json()
        assert body["category"] == "retryable_provider_error"
        assert body["retryable"] is True


def test_groq_configuration_error_guidance_is_provider_specific(monkeypatch, test_settings: Settings):
    import urllib.error

    configured = Settings(**{
        **test_settings.__dict__,
        "llm_api_key": "test",
        "llm_provider": "groq",
        "llm_model": "openai/gpt-oss-20b",
        "llm_base_url": "https://api.groq.com/openai/v1",
        "testing_mode": False,
    })
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://api.groq.com", 403, "Forbidden", {}, None)
        ),
    )
    app = create_app(configured)
    with TestClient(app) as configured_client:
        response = configured_client.post("/api/donations", json={"message": complete_message(configured)})
        assert response.status_code == 503
        assert "GROQ_API_KEY" in response.json()["guidance"]
        assert "openai/gpt-oss-20b" in response.json()["guidance"]


def test_reported_clarification_scenario_persists_items_and_all_answers(test_settings: Settings):
    app = create_app(test_settings)
    with TestClient(app) as first_client:
        created = first_client.post(
            "/api/donations", json={"message": "65 kgs of chicken biryani and 15 litres of raita"}
        ).json()
        donation_id = created["id"]
        assert created["status"] == "NEEDS_CLARIFICATION"
        assert created["safety"] is None
        assert created["match"] is None
        assert [(item["name"], item["quantity"], item["unit"]) for item in created["parsed"]["food_items"]] == [
            ("chicken biryani", 65.0, "kg"),
            ("raita", 15.0, "litres"),
        ]

        payload = {
            "fields": {
                "prepared_at": "6 pm",
                "pickup_deadline": "8 pm",
                "donor_location": "bengaluru",
                "storage_conditions": "refrigerated",
            },
            "idempotency_key": "reported-clarification",
        }
        response = first_client.post(f"/api/donations/{donation_id}/clarify", json=payload)
        assert response.status_code == 200, response.text
        clarified = response.json()
        parsed = clarified["parsed"]
        assert parsed["prepared_at"].endswith("+05:30")
        assert parsed["pickup_deadline"].endswith("+05:30")
        assert parsed["prepared_at"].split("T")[0] == parsed["pickup_deadline"].split("T")[0]
        assert parsed["donor_location"] == "bengaluru"
        assert parsed["storage_conditions"] == "refrigerated"
        assert clarified["original_message"] == "65 kgs of chicken biryani and 15 litres of raita"
        assert parsed["food_items"] == created["parsed"]["food_items"]
        assert clarified["selected_ngo_id"] is None
        assert clarified["attempt_count"] == 0
        assert clarified["match"] is None
        assert clarified["status"] in {"BLOCKED", "REQUIRES_HUMAN_REVIEW"}
        version_after_first = clarified["version"]

        replay = first_client.post(
            f"/api/donations/{donation_id}/clarify",
            json={
                "fields": {**payload["fields"], "donor_location": "should-not-overwrite"},
                "idempotency_key": "reported-clarification",
            },
        )
        assert replay.status_code == 200
        assert replay.json()["version"] == version_after_first
        assert replay.json()["parsed"]["donor_location"] == "bengaluru"

    restarted = create_app(test_settings)
    with TestClient(restarted) as second_client:
        restored = second_client.get(f"/api/donations/{donation_id}")
        assert restored.status_code == 200
        parsed = restored.json()["parsed"]
        assert parsed["donor_location"] == "bengaluru"
        assert parsed["storage_conditions"] == "refrigerated"
        assert parsed["food_items"][0]["quantity"] == 65
        assert parsed["food_items"][1]["quantity"] == 15


def test_invalid_clarification_does_not_mutate_persisted_offer(client: TestClient):
    created = client.post(
        "/api/donations", json={"message": "65 kgs of chicken biryani and 15 litres of raita"}
    ).json()
    donation_id = created["id"]
    invalid = client.post(
        f"/api/donations/{donation_id}/clarify",
        json={"fields": {"prepared_at": "6", "pickup_deadline": "evening"}},
    )
    assert invalid.status_code == 422
    assert "Ambiguous time for prepared_at" in invalid.json()["detail"]
    restored = client.get(f"/api/donations/{donation_id}").json()
    assert restored["version"] == created["version"]
    assert restored["parsed"]["prepared_at"] is None
    assert restored["safety"] is None
    assert restored["match"] is None


def test_empty_clarification_value_is_rejected(client: TestClient):
    created = client.post("/api/donations", json={"message": "Some vegetarian rice is available"}).json()
    response = client.post(
        f"/api/donations/{created['id']}/clarify", json={"fields": {"donor_location": ""}}
    )
    assert response.status_code == 422


def test_workflow_projection_reflects_real_replanning(client: TestClient, test_settings: Settings):
    offer = create_offer(client, test_settings)
    initial = client.get(f"/api/donations/{offer['id']}/workflow")
    assert initial.status_code == 200
    stages = {stage["key"]: stage for stage in initial.json()["stages"]}
    assert stages["intake"]["status"] == "completed"
    assert stages["safety"]["status"] == "completed"
    assert stages["matching"]["status"] == "completed"
    assert stages["approval"]["status"] == "awaiting_human"
    assert stages["intake"]["fallback"] is None
    assert all(stage["duration_ms"] is None or stage["duration_ms"] >= 0 for stage in stages.values())

    client.post(f"/api/donations/{offer['id']}/approve", json={"actor": "demo_donor"})
    client.post(
        f"/api/donations/{offer['id']}/ngo-response",
        json={"actor": "demo_ngo", "response": "decline"},
    )
    replanned = client.get(f"/api/donations/{offer['id']}/workflow").json()
    stages = {stage["key"]: stage for stage in replanned["stages"]}
    assert replanned["current_status"] == "AWAITING_DONOR_APPROVAL"
    assert stages["response"]["status"] == "completed"
    assert stages["replanning"]["status"] == "completed"
    assert stages["approval"]["status"] == "awaiting_human"
    assert "replacement" in stages["replanning"]["explanation"].lower() or "alternative" in stages["replanning"]["explanation"].lower()


def test_demo_scenarios_use_real_workflow_and_audit(client: TestClient):
    complete = client.post(
        "/api/demo/scenarios/complete", json={"idempotency_key": "scenario-complete-key"}
    )
    assert complete.status_code == 201, complete.text
    assert complete.json()["status"] == "AWAITING_DONOR_APPROVAL"
    assert complete.json()["demo_scenario"] == "complete"
    assert any(event["event_type"] == "DEMO_SCENARIO_STARTED" for event in complete.json()["audit_events"])

    missing = client.post(
        "/api/demo/scenarios/missing", json={"idempotency_key": "scenario-missing-key"}
    ).json()
    assert missing["status"] == "NEEDS_CLARIFICATION"
    assert missing["safety"] is None
    assert missing["match"] is None

    safety = client.post(
        "/api/demo/scenarios/safety", json={"idempotency_key": "scenario-safety-key"}
    ).json()
    assert safety["status"] == "REQUIRES_HUMAN_REVIEW"
    assert safety["safety"]["verdict"] == "REQUIRES_HUMAN_REVIEW"
    assert safety["match"] is None

    metrics = client.get("/api/metrics").json()
    assert metrics["TOTAL"] == 3
    assert metrics["AWAITING_CLARIFICATION"] == 1
    assert metrics["BLOCKED_OR_REVIEW"] == 1
    assert metrics["ELIGIBLE_AWAITING_APPROVAL"] == 1


def test_concurrent_creation_with_one_key_produces_one_record(client: TestClient, test_settings: Settings):
    payload = {"message": complete_message(test_settings), "idempotency_key": "concurrent-create-key"}

    def submit():
        response = client.post("/api/donations", json=payload)
        assert response.status_code == 201
        return response.json()["id"]

    with ThreadPoolExecutor(max_workers=6) as executor:
        ids = list(executor.map(lambda _: submit(), range(6)))
    assert len(set(ids)) == 1
    records = client.get("/api/donations").json()
    assert [item["id"] for item in records].count(ids[0]) == 1
    assert len(records) == 1


def test_creation_key_conflict_rejected_but_same_message_with_new_key_allowed(client: TestClient, test_settings: Settings):
    message = complete_message(test_settings)
    first = client.post("/api/donations", json={"message": message, "idempotency_key": "create-key-one"})
    conflict = client.post(
        "/api/donations", json={"message": "A genuinely different synthetic offer", "idempotency_key": "create-key-one"}
    )
    second = client.post("/api/donations", json={"message": message, "idempotency_key": "create-key-two"})
    assert first.status_code == 201
    assert conflict.status_code == 422
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    assert len(client.get("/api/donations").json()) == 2

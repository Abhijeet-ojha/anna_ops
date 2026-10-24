from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


SCENARIO = "annaops_e2e_success"
SAFE_VERDICT = "SAFE_TO_PROCEED_TO_MATCHING"
REQUIRED_AUDIT_EVENTS = {
    "DONATION_CREATED",
    "INTAKE_PARSED",
    "SAFETY_ASSESSED",
    "MATCHING_COMPLETED",
    "DEMO_SCENARIO_STARTED",
    "DONOR_APPROVED_AND_DISPATCHED",
    "NGO_ACCEPTED",
    "DELIVERY_CONFIRMED",
}


class DemoFailure(RuntimeError):
    pass


class ApiFailure(DemoFailure):
    def __init__(self, method: str, path: str, status: int, payload: Any):
        self.status = status
        self.payload = payload
        detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
        category = payload.get("category") if isinstance(payload, dict) else None
        suffix = f" [{category}]" if category else ""
        super().__init__(f"{method} {path} returned HTTP {status}{suffix}: {detail}")


@dataclass
class Stage:
    name: str
    result: str = "NOT RUN"
    persisted_status: str = "-"
    duration_ms: int | None = None
    execution_id: str | None = None
    reason: str = ""


STAGE_NAMES = [
    "Donor submission persisted",
    "Intake Agent structured extraction",
    "Schema and business validation",
    "Deterministic safety checks",
    "Receiver Advisor tool retrieval",
    "Eligibility and receiver selection",
    "Workflow Coordinator decision",
    "Backend transition enforcement",
    "Explicit donor approval",
    "Dispatch Agent request",
    "NGO response evaluates demand",
    "Synthetic NGO acceptance",
    "Delivery confirmation",
    "Final state and audit persistence",
]


class Api:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[Any, int]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return (json.loads(raw) if raw else None), round((time.perf_counter() - started) * 1000)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw[:500]
            raise ApiFailure(method, path, exc.code, payload) from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise DemoFailure(f"Cannot reach {self.base_url}: {type(exc).__name__}: {exc}") from exc

    def get(self, path: str) -> tuple[Any, int]:
        return self.request("GET", path)

    def post(self, path: str, body: dict[str, Any]) -> tuple[Any, int]:
        return self.request("POST", path, body)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoFailure(message)


def execution_for(executions: list[dict[str, Any]], agent_type: str, stage: str | None = None) -> dict[str, Any] | None:
    matches = [item for item in executions if item.get("agent_type") == agent_type]
    if stage:
        matches = [item for item in matches if item.get("workflow_stage") == stage]
    return matches[-1] if matches else None


def mark(
    stages: list[Stage], index: int, donation: dict[str, Any], *, duration_ms: int | None = None,
    execution: dict[str, Any] | None = None, reason: str = ""
) -> None:
    stages[index].result = "PASS"
    stages[index].persisted_status = donation.get("status", "-")
    stages[index].duration_ms = duration_ms if duration_ms is not None else (
        execution.get("duration_ms") if execution else None
    )
    stages[index].execution_id = execution.get("id") if execution else None
    stages[index].reason = reason


def print_report(
    stages: list[Stage], *, provider: str | None, model: str | None, donation: dict[str, Any] | None,
    ngo_name: str | None, failure: str | None
) -> None:
    print("\nAnnaOps live E2E result")
    print("=" * 96)
    print(f"{'#':>2}  {'RESULT':<8} {'STAGE':<39} {'STATE':<25} {'MS':>7}  EXECUTION")
    for number, stage in enumerate(stages, 1):
        duration = "-" if stage.duration_ms is None else str(stage.duration_ms)
        execution = stage.execution_id or "-"
        print(f"{number:>2}  {stage.result:<8} {stage.name:<39} {stage.persisted_status:<25} {duration:>7}  {execution}")
        if stage.reason:
            print(f"    {stage.reason}")

    print("\nRun summary")
    print(f"  Provider/model: {provider or 'unknown'} / {model or 'unknown'}")
    print(f"  Donation ID: {donation.get('id') if donation else 'not created'}")
    print(f"  Selected synthetic NGO: {donation.get('selected_ngo_id') if donation else '-'} / {ngo_name or '-'}")
    usage = (donation or {}).get("llm_usage") or {}
    print(f"  LLM calls: {usage.get('request_count', 'unavailable')}")
    if usage.get("token_metrics_available"):
        print(f"  Token usage: {usage.get('input_tokens', 0)} input / {usage.get('output_tokens', 0)} output")
    else:
        print("  Token usage: unavailable from provider")
    print(f"  Final persisted state: {donation.get('status') if donation else 'NOT RUN'}")
    simulated = bool(((donation or {}).get("final_outcome") or {}).get("synthetic"))
    print(f"  Delivery confirmation simulated: {'yes' if simulated else 'no'}")
    if failure:
        print(f"  Failure: {failure}")


def run(base_url: str, timeout: float) -> int:
    api = Api(base_url, timeout)
    stages = [Stage(name) for name in STAGE_NAMES]
    donation: dict[str, Any] | None = None
    ngo_name: str | None = None
    provider: str | None = None
    model: str | None = None
    failure: str | None = None
    before_ids: set[str] = set()

    try:
        health, _ = api.get("/api/health")
        require(health.get("status") == "ok", f"Application health is {health.get('status')!r}")
        require(health.get("llm") == "configured", "No live LLM provider is configured")
        require(health.get("llm_api_key_present") is True, "Configured provider API key is absent")
        provider, model = health.get("llm_provider"), health.get("llm_model")
        print(f"Configuration: PASS ({provider} / {model}); API key present (value redacted).")

        before_donations, _ = api.get("/api/donations")
        before_ids = {item["id"] for item in before_donations}
        before_ngos, _ = api.get("/api/ngos")
        active_vegetarian_demand = any(
            ngo.get("available")
            and ngo.get("available_capacity", 0) >= 25
            and any(
                demand.get("active")
                and demand.get("unit") == "portions"
                and demand.get("remaining_quantity", 0) >= 25
                and demand.get("minimum_quantity", 0) <= 25
                and demand.get("item_name") == "vegetarian food"
                for demand in ngo.get("demands", [])
            )
            for ngo in before_ngos
        )
        require(active_vegetarian_demand, "No seeded synthetic NGO has active capacity and demand for 25 vegetarian portions")
        print("Synthetic demand: PASS (idempotently seeded NGO demand is available).")

        run_id = uuid.uuid4().hex
        try:
            donation, submit_ms = api.post(
                f"/api/demo/scenarios/{SCENARIO}",
                {"idempotency_key": f"{SCENARIO}-{run_id}"},
            )
        except Exception:
            latest, _ = api.get("/api/donations")
            created = [item for item in latest if item["id"] not in before_ids]
            if created:
                donation, _ = api.get(f"/api/donations/{created[0]['id']}")
            raise

        mark(stages, 0, donation, duration_ms=submit_ms)
        require(donation["status"] == "AWAITING_DONOR_APPROVAL", f"Expected AWAITING_DONOR_APPROVAL, got {donation['status']}")

        executions = donation.get("agent_executions", [])
        intake = execution_for(executions, "intake_agent", "intake")
        require(intake is not None and intake.get("execution_mode") == "llm" and not intake.get("error"), "Intake Agent did not complete through the live LLM")
        require((donation.get("parsed") or {}).get("extraction_source") == "configured_llm", "Persisted intake source is not configured_llm")
        mark(stages, 1, donation, execution=intake)

        parsed = donation.get("parsed") or {}
        require(not parsed.get("missing_fields") and not parsed.get("contradictions"), "Validated offer still has missing or contradictory facts")
        require(bool(parsed.get("food_items")), "Validated offer has no structured food items")
        mark(stages, 2, donation, reason="Pydantic output persisted with no missing fields or contradictions.")

        safety = execution_for(executions, "safety_agent", "safety")
        require((donation.get("safety") or {}).get("verdict") == SAFE_VERDICT, "Authoritative safety verdict did not permit matching")
        mark(stages, 3, donation, execution=safety)

        advisor = execution_for(executions, "receiver_advisor_agent", "matching")
        tool_names = {event.get("name") for event in (advisor or {}).get("tools", [])}
        require("run_eligibility_and_scoring" in tool_names, "Receiver Advisor did not run its registered eligibility/scoring tool")
        scoring_event = next(event for event in (advisor or {}).get("tools", []) if event.get("name") == "run_eligibility_and_scoring")
        require(isinstance(scoring_event.get("result"), dict), "Receiver Advisor scoring tool did not return persisted receiver evidence")
        scoring_arguments = scoring_event.get("arguments") or {}
        require(scoring_event.get("argument_source") == "backend_bound", "Receiver Advisor did not label protected backend-bound inputs")
        require(bool(scoring_arguments.get("donation_id")), "Receiver Advisor tool audit is missing the donation ID")
        require(bool(scoring_arguments.get("offered_food")), "Receiver Advisor tool audit is missing the offered food")
        require(bool(scoring_arguments.get("storage_conditions")), "Receiver Advisor tool audit is missing storage conditions")
        mark(stages, 4, donation, execution=advisor, reason="Registered scoring tool exposed readable, backend-bound inputs and persisted results.")

        require(bool(donation.get("selected_ngo_id")), "No receiver was selected")
        candidates = (donation.get("match") or {}).get("candidates", [])
        require(any(item.get("ngo_id") == donation["selected_ngo_id"] for item in candidates), "Selected receiver is not in the authoritative candidate set")
        mark(stages, 5, donation, execution=advisor)

        coordinator = execution_for(executions, "workflow_coordinator", "coordination")
        require(coordinator is not None and coordinator.get("selected_action") == "run_intake", "Workflow Coordinator did not propose the authoritative initial operation")
        state_tool = next((event for event in coordinator.get("tools", []) if event.get("name") == "read_persisted_workflow_state"), None)
        require(bool((state_tool or {}).get("arguments", {}).get("donation_id")), "Workflow Coordinator tool audit is missing its bound donation ID")
        mark(stages, 6, donation, execution=coordinator)

        audit_types = [event.get("event_type") for event in donation.get("audit_events", [])]
        required_prefix = ["DONATION_CREATED", "INTAKE_PARSED", "SAFETY_ASSESSED", "MATCHING_COMPLETED", "DEMO_SCENARIO_STARTED"]
        require(all(item in audit_types for item in required_prefix), "Initial transition audit is incomplete")
        mark(stages, 7, donation, reason="No mandatory stage was skipped in the persisted transition audit.")

        selected_id = donation["selected_ngo_id"]
        before_selected = next(item for item in before_ngos if item["id"] == selected_id)
        ngo_name = before_selected["name"]

        donation, approve_ms = api.post(
            f"/api/donations/{donation['id']}/approve",
            {"actor": "demo_donor", "reason": "Authorized synthetic E2E donor approval", "idempotency_key": f"approve-{run_id}"},
        )
        require(donation["status"] == "AWAITING_NGO_RESPONSE", "Approval did not create the simulated dispatch")
        require(any(item.get("decision") == "APPROVED" for item in donation.get("approval_decisions", [])), "Donor approval was not persisted")
        mark(stages, 8, donation, duration_ms=approve_ms)

        dispatch_exec = execution_for(donation.get("agent_executions", []), "dispatch_agent", "dispatch")
        require(any(item.get("status") == "AWAITING_RESPONSE" for item in donation.get("dispatch_attempts", [])), "No pending dispatch attempt was persisted")
        mark(stages, 9, donation, execution=dispatch_exec)

        donation, response_ms = api.post(
            f"/api/donations/{donation['id']}/ngo-response",
            {"actor": "demo_ngo", "response": "accept", "reason": "Deterministic synthetic acceptance after eligibility revalidation", "idempotency_key": f"accept-{run_id}"},
        )
        require(donation["status"] == "ACCEPTED", "Synthetic receiver did not accept the dispatch")
        accepted_event = next((event for event in donation.get("audit_events", []) if event.get("event_type") == "NGO_ACCEPTED"), None)
        reservation = ((accepted_event or {}).get("details") or {}).get("demand_reservation")
        require(bool(reservation), "NGO acceptance did not persist an evaluated demand reservation")
        mark(stages, 10, donation, duration_ms=response_ms, reason="Atomic demand/capacity reservation persisted.")

        response_exec = execution_for(donation.get("agent_executions", []), "dispatch_agent", "response")
        require(any(item.get("status") == "ACCEPT" for item in donation.get("dispatch_attempts", [])), "Accepted NGO response was not persisted")
        mark(stages, 11, donation, execution=response_exec)

        donation, delivery_ms = api.post(
            f"/api/donations/{donation['id']}/delivery-confirmation",
            {"actor": "demo_coordinator", "verification_note": "Synthetic delivery confirmed by the AnnaOps E2E runner", "idempotency_key": f"delivery-{run_id}"},
        )
        require(donation["status"] == "COMPLETED" and donation.get("delivery_confirmed_at"), "Delivery confirmation did not complete the donation")
        completion_exec = execution_for(donation.get("agent_executions", []), "dispatch_agent", "completion")
        mark(stages, 12, donation, duration_ms=delivery_ms, execution=completion_exec)

        persisted, _ = api.get(f"/api/donations/{donation['id']}")
        audit, _ = api.get(f"/api/donations/{donation['id']}/audit")
        workflow, _ = api.get(f"/api/donations/{donation['id']}/workflow")
        after_ngos, _ = api.get("/api/ngos")
        after_selected = next(item for item in after_ngos if item["id"] == selected_id)
        actual_events = {event.get("event_type") for event in audit}
        require(persisted["status"] == "COMPLETED" and workflow["current_status"] == "COMPLETED", "Independent reads did not return COMPLETED")
        require(REQUIRED_AUDIT_EVENTS <= actual_events, f"Audit events missing: {sorted(REQUIRED_AUDIT_EVENTS - actual_events)}")
        offered_quantity = sum(float(item["quantity"]) for item in parsed["food_items"])
        capacity_delta = float(before_selected["available_capacity"]) - float(after_selected["available_capacity"])
        require(abs(capacity_delta - offered_quantity) < 0.001, f"Receiver capacity changed by {capacity_delta}, expected {offered_quantity}")
        require(persisted.get("final_outcome", {}).get("delivery") == "confirmed_in_demo", "Final outcome is missing simulated delivery confirmation")
        donation = persisted
        mark(stages, 13, donation, reason=f"Audit complete; receiver capacity decreased by {capacity_delta:g}.")

    except Exception as exc:
        failure = str(exc)
        first_not_run = next((item for item in stages if item.result == "NOT RUN"), None)
        if first_not_run:
            first_not_run.result = "FAIL"
            first_not_run.persisted_status = donation.get("status", "-") if donation else "-"
            first_not_run.reason = failure
        if donation and donation.get("id"):
            try:
                donation, _ = api.get(f"/api/donations/{donation['id']}")
            except Exception:
                pass

    print_report(stages, provider=provider, model=model, donation=donation, ngo_name=ngo_name, failure=failure)
    return 1 if failure else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real, synthetic AnnaOps workflow through the local API.")
    parser.add_argument("--scenario", default=SCENARIO, choices=[SCENARIO])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    return run(args.base_url, args.timeout)


if __name__ == "__main__":
    sys.exit(main())

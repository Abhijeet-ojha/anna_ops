from __future__ import annotations

import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from annaops import __version__
from annaops.agents.dispatch import DispatchAgent, InvalidTransition
from annaops.agents.matching import FICTIONAL_NGOS
from annaops.config import Settings, settings
from annaops.db import Database
from annaops.schemas import (
    ActionRequest,
    ClarificationRequest,
    DemoScenarioRequest,
    DeliveryConfirmationRequest,
    DonationCreate,
    NGOResponseRequest,
)
from annaops.workflow import DonationWorkflow
from annaops.projections import project_workflow


def create_app(config: Settings = settings) -> FastAPI:
    database = Database(config)
    workflow = DonationWorkflow(database, config)
    dispatch = DispatchAgent(database, config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        database.seed_ngos(FICTIONAL_NGOS)
        yield

    app = FastAPI(
        title="AnnaOps",
        version=__version__,
        description="Local synthetic demo for surplus-food donation coordination.",
        lifespan=lifespan,
    )
    app.state.database = database
    app.state.workflow = workflow
    app.state.dispatch = dispatch

    @app.exception_handler(KeyError)
    async def not_found_handler(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": f"Donation {exc.args[0]} was not found"})

    @app.exception_handler(InvalidTransition)
    async def transition_handler(_, exc: InvalidTransition):
        return JSONResponse(status_code=409, content={"error": "invalid_transition", "detail": str(exc)})

    @app.exception_handler(RuntimeError)
    async def runtime_handler(_, exc: RuntimeError):
        return JSONResponse(status_code=409, content={"error": "workflow_conflict", "detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_handler(_, exc: ValueError):
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        try:
            database.metrics()
            database_status = "ok"
        except Exception:
            database_status = "error"
        llm_configured = bool(config.llm_api_key and config.llm_model)
        llm_provider = "gemini" if "generativelanguage.googleapis.com" in config.llm_base_url else "openai_compatible"
        return {
            "status": "ok" if database_status == "ok" else "degraded",
            "version": __version__,
            "database": database_status,
            "demo_mode": config.demo_mode,
            "llm": "configured" if llm_configured else "deterministic_demo_fallback",
            "llm_provider": llm_provider if llm_configured else None,
            "llm_model": config.llm_model if llm_configured else None,
            "agent_runtime": {"max_iterations": config.agent_max_iterations, "timeout_seconds": config.llm_timeout_seconds},
            "routing": "configured_osrm" if config.routing_base_url else "straight_line_fallback",
            "warning": "Synthetic local demo; no real NGOs or delivery services are connected.",
        }

    @app.get("/api/metrics")
    def metrics() -> dict[str, int]:
        return database.metrics()

    @app.get("/api/ngos")
    def ngos() -> list[dict[str, Any]]:
        return database.list_ngos()

    @app.post("/api/donations", status_code=201)
    def create_donation(body: DonationCreate) -> dict[str, Any]:
        return _create_and_process(body.message, body.idempotency_key)

    def _create_and_process(
        message: str, idempotency_key: str | None, scenario: str | None = None
    ) -> dict[str, Any]:
        donation, created = database.create_donation_once(
            message, "INTAKE_PROCESSING", idempotency_key, synthetic=config.demo_mode
        )
        if not created:
            return donation
        try:
            result = workflow.process_new(donation["id"], message)
        except Exception as exc:
            current = database.get_donation(donation["id"])
            database.update_donation(
                donation["id"], current["status"], "ERROR", "WORKFLOW_ERROR", "system",
                "Processing failed without overwriting the last persisted facts",
                {"error_type": type(exc).__name__}, error_message=str(exc),
            )
            raise HTTPException(status_code=500, detail="Offer processing failed safely") from exc
        if scenario:
            database.audit(
                donation["id"], "DEMO_SCENARIO_STARTED", result["status"], result["status"],
                {"scenario": scenario, "synthetic": True}, "demo_operator",
                f"Synthetic '{scenario}' judge scenario launched",
            )
            result = database.get_donation(donation["id"])
        return result

    @app.post("/api/demo/scenarios/{scenario}", status_code=201)
    def run_demo_scenario(scenario: str, body: DemoScenarioRequest) -> dict[str, Any]:
        now = datetime.now(config.timezone).replace(microsecond=0)
        prepared = (now - timedelta(minutes=20)).isoformat()
        deadline = (now + timedelta(hours=3)).isoformat()
        scenarios = {
            "complete": (
                f"40 portions vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "missing": "We have surplus meals left after a synthetic college event.",
            "rejection": (
                f"35 portions vegetarian rice prepared at {prepared}, refrigerated, "
                f"pickup by {deadline} from Demo Hall Jaipur, coordinates 26.9050, 75.7750"
            ),
            "safety": (
                f"24 portions vegetarian rice prepared at {prepared}, room temperature, "
                f"pickup by {deadline} from Demo Hall Jaipur, coordinates 26.9050, 75.7750"
            ),
            "escalation": (
                f"30 portions vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Kitchen Jaipur, coordinates 26.9100, 75.7800"
            ),
        }
        if scenario not in scenarios:
            raise HTTPException(status_code=404, detail="Unknown demo scenario")
        result = _create_and_process(scenarios[scenario], body.idempotency_key, scenario=scenario)
        result["demo_scenario"] = scenario
        result["demo_guidance"] = {
            "complete": "Approve the proposal, simulate acceptance, then record delivery separately.",
            "missing": "Supply the displayed missing facts; the workflow will resume from validation.",
            "rejection": "Approve, simulate decline, inspect the replacement, then approve again.",
            "safety": "Inspect the deterministic review verdict; matching and dispatch remain blocked.",
            "escalation": "Approve and simulate two timeouts or declines to reach bounded escalation.",
        }[scenario]
        return result

    @app.get("/api/donations")
    def list_donations(
        status: str | None = Query(default=None), search: str | None = Query(default=None, max_length=200)
    ) -> list[dict[str, Any]]:
        return database.list_donations(status=status, search=search)

    @app.get("/api/donations/{donation_id}")
    def get_donation(donation_id: str) -> dict[str, Any]:
        return database.get_donation(donation_id)

    @app.get("/api/donations/{donation_id}/audit")
    def get_audit(donation_id: str) -> list[dict[str, Any]]:
        return database.get_donation(donation_id)["audit_events"]

    @app.get("/api/donations/{donation_id}/workflow")
    def get_workflow(donation_id: str) -> dict[str, Any]:
        return project_workflow(database.get_donation(donation_id))

    @app.get("/api/donations/{donation_id}/agent-executions")
    def get_agent_executions(donation_id: str) -> list[dict[str, Any]]:
        database.get_donation(donation_id)
        return database.list_agent_executions(donation_id)

    @app.post("/api/donations/{donation_id}/clarify")
    def clarify(donation_id: str, body: ClarificationRequest) -> dict[str, Any]:
        cached = database.idempotent_response(donation_id, "clarify", body.idempotency_key)
        if cached:
            return cached
        result = workflow.process_clarification(donation_id, body.fields)
        database.save_idempotent_response(donation_id, "clarify", body.idempotency_key, result)
        return result

    @app.post("/api/donations/{donation_id}/approve")
    def approve(donation_id: str, body: ActionRequest) -> dict[str, Any]:
        _require_demo_actor(body.actor, "donor")
        return _idempotent_action(database, donation_id, "approve", body.idempotency_key,
                                  lambda: dispatch.approve(donation_id, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/reject")
    def reject(donation_id: str, body: ActionRequest) -> dict[str, Any]:
        _require_demo_actor(body.actor, "donor")
        return _idempotent_action(database, donation_id, "reject", body.idempotency_key,
                                  lambda: dispatch.reject_proposal(donation_id, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/ngo-response")
    def ngo_response(donation_id: str, body: NGOResponseRequest) -> dict[str, Any]:
        _require_demo_actor(body.actor, "ngo")
        return _idempotent_action(database, donation_id, f"ngo-{body.response}", body.idempotency_key,
                                  lambda: dispatch.respond(donation_id, body.response, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/delivery-confirmation")
    def delivery_confirmation(donation_id: str, body: DeliveryConfirmationRequest) -> dict[str, Any]:
        _require_demo_actor(body.actor, "coordinator")
        return _idempotent_action(database, donation_id, "delivery", body.idempotency_key,
                                  lambda: dispatch.confirm_delivery(donation_id, body.actor, body.verification_note))

    @app.post("/api/donations/{donation_id}/escalate")
    def escalate(donation_id: str, body: ActionRequest) -> dict[str, Any]:
        _require_demo_actor(body.actor, "coordinator")
        return _idempotent_action(database, donation_id, "escalate", body.idempotency_key,
                                  lambda: dispatch.escalate(donation_id, body.actor, body.reason))

    @app.get("/api/donations/{donation_id}/record.csv")
    def donation_record(donation_id: str):
        donation = database.get_donation(donation_id)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["AnnaOps synthetic donation record"])
        writer.writerow(["This record documents a local simulation; it is not proof of a real-world delivery."])
        writer.writerow([])
        writer.writerow(["field", "value"])
        for field in (
            "id", "status", "original_message", "selected_ngo_id", "attempt_count",
            "donor_approved", "delivery_confirmed_at", "created_at", "updated_at", "synthetic",
        ):
            writer.writerow([field, donation.get(field)])
        parsed = donation.get("parsed") or {}
        for field in ("food_type", "food_category", "quantity", "unit", "prepared_at", "pickup_deadline", "storage_conditions", "donor_location"):
            writer.writerow([f"offer.{field}", parsed.get(field)])
        writer.writerow(["final_outcome", donation.get("final_outcome")])
        response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment; filename="annaops-{donation_id}.csv"'
        return response

    static_dir = Path(__file__).resolve().parent.parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    return app


def _require_demo_actor(actor: str, role: str) -> None:
    if role not in actor.lower():
        raise HTTPException(
            status_code=403,
            detail=f"Local demo role gate requires an actor label containing '{role}'. This is not production authentication.",
        )


def _idempotent_action(database: Database, donation_id: str, action: str, key: str | None, operation):
    cached = database.idempotent_response(donation_id, action, key)
    if cached:
        return cached
    result = operation()
    database.save_idempotent_response(donation_id, action, key, result)
    return result


app = create_app()


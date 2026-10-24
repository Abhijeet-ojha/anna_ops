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
from annaops.agents.runtime import AgentExecutionLimitError, AgentProtocolError
from annaops.config import CANONICAL_MODEL, ConfigurationError, Settings, settings
from annaops.db import BudgetExceeded, Database
from annaops.llm import (
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
    LLMRetryableError,
    StructuredLLMClient,
    TestMockClient,
)
from annaops.schemas import (
    ActionRequest,
    ClarificationRequest,
    DeliveryConfirmationRequest,
    DemoScenarioRequest,
    DonationCreate,
    NGOResponseRequest,
    SafetyReviewRequest,
)
from annaops.workflow import DonationWorkflow
from annaops.projections import project_workflow


def create_app(config: Settings = settings, llm_client: StructuredLLMClient | Any | None = None) -> FastAPI:
    if config.testing_mode and llm_client is None:
        llm_client = TestMockClient(config)

    database = Database(config)
    database.initialize()
    workflow = DonationWorkflow(database, config, client=llm_client)
    dispatch = DispatchAgent(database, config, client=llm_client)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Validate the configured LLM provider at application startup
        config.validate()
        database.initialize()
        database.seed_ngos(FICTIONAL_NGOS)
        yield

    app = FastAPI(
        title="AnnaOps",
        version=__version__,
        description="Surplus-food donation coordination powered by the configured LLM provider.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.database = database
    app.state.workflow = workflow
    app.state.dispatch = dispatch

    @app.exception_handler(KeyError)
    async def not_found_handler(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"error": "not_found", "detail": f"Donation {exc.args[0]} was not found"})

    @app.exception_handler(ValueError)
    async def value_error_handler(_, exc: ValueError):
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": str(exc)})

    @app.exception_handler(InvalidTransition)
    async def transition_handler(_, exc: InvalidTransition):
        return JSONResponse(status_code=409, content={"error": "invalid_transition", "detail": str(exc)})

    @app.exception_handler(ConfigurationError)
    async def config_error_handler(_, exc: ConfigurationError):
        return JSONResponse(status_code=503, content={
            "error": "configuration_error",
            "detail": str(exc),
            "category": "configuration_error",
            "retryable": False,
        })

    @app.exception_handler(LLMConfigurationError)
    async def llm_config_handler(_, exc: LLMConfigurationError):
        provider = config.llm_provider
        model = config.llm_model or "configured model"
        key_name = "GROQ_API_KEY" if provider == "groq" else "ANNAOPS_GEMINI_API_KEY"
        return JSONResponse(status_code=503, content={
            "error": "configuration_error",
            "detail": str(exc),
            "category": exc.category,
            "retryable": False,
            "guidance": f"Verify that {key_name} is configured with access to {model}.",
        })

    @app.exception_handler(LLMRetryableError)
    async def llm_retryable_handler(_, exc: LLMRetryableError):
        return JSONResponse(status_code=503, content={
            "error": "provider_unavailable",
            "detail": str(exc),
            "category": exc.category,
            "retryable": True,
            "guidance": "The LLM provider is temporarily unavailable or rate-limited. Retry when provider recovers.",
        })

    @app.exception_handler(LLMResponseError)
    async def llm_response_handler(_, exc: LLMResponseError):
        return JSONResponse(status_code=502, content={
            "error": "invalid_model_output",
            "detail": str(exc),
            "category": exc.category,
            "retryable": True,
            "guidance": "The model returned an invalid structured output. Retry the operation.",
        })

    @app.exception_handler(AgentExecutionLimitError)
    async def agent_limit_handler(_, exc: AgentExecutionLimitError):
        return JSONResponse(status_code=502, content={
            "error": "agent_execution_limit_failure",
            "detail": str(exc),
            "category": exc.category,
            "retryable": True,
            "guidance": "Agent reached the maximum iteration limit without a valid decision. Retry the operation.",
        })

    @app.exception_handler(AgentProtocolError)
    async def agent_protocol_handler(_, exc: AgentProtocolError):
        return JSONResponse(status_code=502, content={
            "error": "agent_protocol_failure",
            "detail": str(exc),
            "category": getattr(exc, "category", "agent_protocol_failure"),
            "retryable": True,
            "guidance": "An agent attempted a forbidden action or unknown tool. Retry the operation.",
        })

    @app.exception_handler(RuntimeError)
    async def runtime_handler(_, exc: RuntimeError):
        return JSONResponse(status_code=409, content={"error": "workflow_conflict", "detail": str(exc)})

    @app.exception_handler(BudgetExceeded)
    async def budget_handler(_, exc: BudgetExceeded):
        return JSONResponse(status_code=409, content={
            "error": "llm_budget_exhausted",
            "detail": str(exc),
            "category": exc.category,
            "retryable": False,
            "guidance": "Extend the audited donation budget or retry after coordinator intervention.",
        })

    @app.get("/api/health")
    def health():
        try:
            database.metrics()
            database_status = "ok"
        except Exception:
            database_status = "error"
        llm_configured = bool(config.llm_api_key and config.llm_model and config.llm_provider)
        return {
            "status": "ok" if database_status == "ok" else "degraded",
            "version": __version__,
            "database": database_status,
            "demo_mode": config.demo_mode,
            "llm": "configured" if llm_configured else "not_configured",
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "llm_base_url": config.llm_base_url,
            "llm_api_key_present": bool(config.llm_api_key),
            "llm_config_source": "GROQ_API_KEY/GROQ_MODEL/GROQ_BASE_URL" if config.llm_provider == "groq"
            else "ANNAOPS_GEMINI_API_KEY/ANNAOPS_GEMINI_MODEL/ANNAOPS_GEMINI_BASE_URL",
            "canonical_model": CANONICAL_MODEL if config.llm_provider == "gemini" else None,
            "agent_runtime": {"max_iterations": config.agent_max_iterations, "timeout_seconds": config.llm_timeout_seconds},
            "llm_controls": {
                "budget_per_donation": config.llm_budget_per_donation,
                "max_retries": config.llm_max_retries,
                "concurrency_limit": config.llm_concurrency_limit,
            },
            "routing": "configured_osrm" if config.routing_base_url else "straight_line_fallback",
            "warning": "Synthetic local demo; no real NGOs or delivery services are connected.",
        }

    @app.get("/api/metrics")
    def metrics():
        return database.metrics()

    @app.get("/api/ngos")
    def list_ngos():
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
            if current["status"] != "ERROR":
                database.update_donation(
                    donation["id"], current["status"], "ERROR", "WORKFLOW_ERROR", "system",
                    "Processing failed without overwriting the last persisted facts",
                    {"error_type": type(exc).__name__}, error_message=str(exc),
                )
            if isinstance(exc, (LLMError, AgentProtocolError, BudgetExceeded)):
                raise
            raise HTTPException(status_code=500, detail="Offer processing failed safely") from exc

        if scenario:
            current_status = result["status"] if isinstance(result, dict) and "status" in result else donation["status"]
            database.audit(
                donation["id"], "DEMO_SCENARIO_STARTED", current_status, current_status,
                {"scenario": scenario, "synthetic": True}, "demo_operator",
                f"Synthetic '{scenario}' judge scenario launched",
            )
            result = database.get_donation(donation["id"])

        return result

    @app.post("/api/donations/{donation_id}/retry")
    def retry_donation(donation_id: str, body: ActionRequest | None = None):
        """Authorised retry of a donation in ERROR status."""
        actor = body.actor if body else "demo_coordinator"
        _require_demo_actor(actor, "coordinator")
        key = body.idempotency_key if body else None
        cached = database.idempotent_response(donation_id, "retry", key)
        if cached:
            return cached
        try:
            result = workflow.retry_processing(donation_id)
        except Exception as exc:
            if isinstance(exc, (LLMError, AgentProtocolError, BudgetExceeded)):
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        database.save_idempotent_response(donation_id, "retry", key, result)
        return result

    def _execute_scenario(scenario_id: str, idempotency_key: str):
        now = datetime.now(config.timezone).replace(microsecond=0)
        prepared = (now - timedelta(minutes=20)).isoformat()
        deadline = (now + timedelta(hours=3)).isoformat()
        scenarios = {
            "annaops_e2e_success": (
                f"25 portions synthetic vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "complete": (
                f"40 portions vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "complete_veg": (
                f"40 portions vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "demand_pasta": (
                f"30 kg pasta prepared at {prepared}, kept hot, pickup by {deadline} "
                f"from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "demand_incompatible": (
                f"10 kg sushi prepared at {prepared}, kept hot, pickup by {deadline} "
                f"from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "demand_quantity_mismatch": (
                f"10 kg pasta prepared at {prepared}, kept hot, pickup by {deadline} "
                f"from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
            "missing": "We have surplus meals left after a synthetic college event.",
            "clarification": "We have surplus meals left after a synthetic college event.",
            "rejection": (
                f"35 portions vegetarian rice prepared at {prepared}, refrigerated, "
                f"pickup by {deadline} from Demo Hall Jaipur, coordinates 26.9050, 75.7750"
            ),
            "safety": (
                f"24 portions vegetarian rice prepared at {prepared}, room temperature, "
                f"pickup by {deadline} from Demo Hall Jaipur, coordinates 26.9050, 75.7750"
            ),
            "blocked": (
                f"24 portions vegetarian rice prepared at {prepared}, room temperature, "
                f"pickup by {deadline} from Demo Hall Jaipur, coordinates 26.9050, 75.7750"
            ),
            "escalation": (
                f"30 portions vegetarian biryani prepared at {prepared}, kept hot, "
                f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
            ),
        }
        if scenario_id not in scenarios:
            raise HTTPException(status_code=404, detail="Unknown demo scenario")

        result = _create_and_process(scenarios[scenario_id], idempotency_key, scenario=scenario_id)
        result["demo_scenario"] = scenario_id
        result["demo_guidance"] = {
            "annaops_e2e_success": "Use the E2E runner to approve, accept, confirm delivery, and verify persistence.",
            "complete": "Approve the proposal, simulate acceptance, then record delivery separately.",
            "complete_veg": "Approve the proposal, simulate acceptance, then record delivery separately.",
            "demand_pasta": "A persisted synthetic pasta demand should produce an explainable receiver proposal.",
            "demand_incompatible": "No synthetic demand accepts this item; the workflow must remain without a receiver.",
            "demand_quantity_mismatch": "The quantity is below the configured demand minimum; inspect exclusions.",
            "missing": "Supply the displayed missing facts; the workflow will resume from validation.",
            "clarification": "Supply the displayed missing facts; the workflow will resume from validation.",
            "rejection": "Approve, simulate decline, inspect the replacement, then approve again.",
            "safety": "Inspect the deterministic review verdict; matching and dispatch remain blocked.",
            "blocked": "Inspect the deterministic review verdict; matching and dispatch remain blocked.",
            "escalation": "Approve and simulate two timeouts or declines to reach bounded escalation.",
        }.get(scenario_id, "Demo scenario initiated.")
        return result

    @app.post("/api/demo/scenario/{scenario_id}", status_code=201)
    def run_scenario(scenario_id: str, body: DemoScenarioRequest):
        return _execute_scenario(scenario_id, body.idempotency_key)

    @app.post("/api/demo/scenarios/{scenario_id}", status_code=201)
    def run_scenarios(scenario_id: str, body: DemoScenarioRequest):
        return _execute_scenario(scenario_id, body.idempotency_key)

    @app.get("/api/donations")
    def list_donations(status: str | None = Query(default=None), search: str | None = Query(default=None)):
        return database.list_donations(status=status, search=search)

    @app.get("/api/donations/{donation_id}")
    def get_donation(donation_id: str):
        return database.get_donation(donation_id)

    @app.get("/api/donations/{donation_id}/audit")
    def get_audit(donation_id: str):
        return database.get_donation(donation_id)["audit_events"]

    @app.get("/api/donations/{donation_id}/workflow")
    def get_workflow(donation_id: str):
        return project_workflow(database.get_donation(donation_id))

    @app.get("/api/donations/{donation_id}/agent-executions")
    def get_agent_executions(donation_id: str):
        database.get_donation(donation_id)
        return database.list_agent_executions(donation_id)

    @app.post("/api/donations/{donation_id}/llm-budget-extension")
    def extend_llm_budget(donation_id: str, body: ActionRequest):
        _require_demo_actor(body.actor, "coordinator")
        donation = database.get_donation(donation_id)
        extension = database.extend_llm_budget(
            donation_id, body.actor, body.reason or "Coordinator approved recovery budget",
            config.llm_budget_extension,
        )
        return {**donation, "llm_budget_extension": extension}

    @app.post("/api/donations/{donation_id}/clarify")
    def clarify(donation_id: str, body: ClarificationRequest):
        cached = database.idempotent_response(donation_id, "clarify", body.idempotency_key)
        if cached:
            return cached
        try:
            result = workflow.submit_clarification(donation_id, body.fields)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        database.save_idempotent_response(donation_id, "clarify", body.idempotency_key, result)
        return result

    @app.post("/api/donations/{donation_id}/safety-review")
    def safety_review(donation_id: str, body: SafetyReviewRequest):
        _require_demo_actor(body.actor, "coordinator")
        cached = database.idempotent_response(donation_id, "safety-review", body.idempotency_key)
        if cached:
            return cached
        try:
            result = workflow.review_safety(donation_id, body.decision, body.actor, body.reason)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        database.save_idempotent_response(donation_id, "safety-review", body.idempotency_key, result)
        return result

    @app.post("/api/donations/{donation_id}/approve")
    def approve(donation_id: str, body: ActionRequest):
        _require_demo_actor(body.actor, "donor")
        return _idempotent_action(database, donation_id, "approve", body.idempotency_key,
                                  lambda: dispatch.approve(donation_id, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/reject")
    def reject(donation_id: str, body: ActionRequest):
        _require_demo_actor(body.actor, "donor")
        return _idempotent_action(database, donation_id, "reject", body.idempotency_key,
                                  lambda: dispatch.reject_proposal(donation_id, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/ngo-response")
    def ngo_response(donation_id: str, body: NGOResponseRequest):
        _require_demo_actor(body.actor, "ngo")
        return _idempotent_action(database, donation_id, f"ngo-{body.response}", body.idempotency_key,
                                  lambda: dispatch.respond(donation_id, body.response, body.actor, body.reason))

    @app.post("/api/donations/{donation_id}/delivery-confirmation")
    def delivery_confirmation(donation_id: str, body: DeliveryConfirmationRequest):
        _require_demo_actor(body.actor, "coordinator")
        return _idempotent_action(database, donation_id, "delivery", body.idempotency_key,
                                  lambda: dispatch.confirm_delivery(donation_id, body.actor, body.verification_note))

    @app.post("/api/donations/{donation_id}/escalate")
    def escalate(donation_id: str, body: ActionRequest):
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

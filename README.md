# अन्न ops

Local-first surplus-food donation coordination with a FastAPI backend, SQLite
persistence, a LangGraph workflow, and a small browser dashboard.

> **Simulation only:** NGOs, locations, routes, dispatches, and deliveries are
> synthetic. This project is not a food-safety certification or production
> donation service.

## What it does

1. Accepts a donor's natural-language offer.
2. Uses the configured OpenAI-compatible LLM to extract a validated offer.
3. Requests clarification when required facts are missing or contradictory.
4. Applies deterministic, versioned food-safety rules.
5. Uses persisted synthetic receiver demand and capacity to rank eligible NGOs.
6. Collects donor approval, simulates NGO responses, retries or escalates failed
   dispatches, and records each transition in SQLite.

The LLM proposes bounded actions; backend code remains authoritative for
validation, safety, permissions, idempotency, budgets, and state transitions.
LLM failures are explicit and persisted as `ERROR` rather than silently falling
back to a different decision path.

## Architecture

| Area | Location | Responsibility |
| --- | --- | --- |
| HTTP API and dashboard hosting | `annaops/main.py` | FastAPI routes, lifecycle, error responses, static UI |
| Workflow orchestration | `annaops/workflow.py` | LangGraph intake → safety → matching flow |
| Agent implementations | `annaops/agents/` | Intake, coordinator, receiver advice, dispatch, safety, runtime |
| Domain models | `annaops/schemas.py` | Pydantic requests, offers, matches, decisions, statuses |
| Persistence | `annaops/db.py` | SQLite schema, transactions, audit events, budgets, idempotency |
| Provider client | `annaops/llm.py` | OpenAI-compatible structured LLM calls and bounded retries |
| Frontend | `static/` | Agent Command Centre dashboard |
| Synthetic seed data | `annaops/seed.py` | Demo NGOs, demand, and scenarios |

## Requirements

- Python 3.11+
- One OpenAI-compatible provider:
  - **Groq** (selected when `GROQ_API_KEY` is present), or
  - **Gemini** (`ANNAOPS_GEMINI_API_KEY`)

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` with one provider. Groq is the default example:

```env
GROQ_API_KEY=your_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Or use Gemini when GROQ_API_KEY is absent:
# ANNAOPS_GEMINI_API_KEY=your_key
# ANNAOPS_GEMINI_MODEL=gemini-3.6-flash
# ANNAOPS_GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

Start the service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn annaops.main:app --env-file .env --port 8000
```

Open the dashboard at <http://127.0.0.1:8000> or API docs at
<http://127.0.0.1:8000/docs>.

Seed or reset synthetic data with:

```powershell
.\.venv\Scripts\python.exe -m annaops
```

## Workflow and API

The main path is:

`INTAKE_PROCESSING` → `NEEDS_CLARIFICATION` or `PENDING_SAFETY_REVIEW` →
`MATCHING` → `AWAITING_DONOR_APPROVAL` → `AWAITING_NGO_RESPONSE` →
`ACCEPTED` → `COMPLETED`

Safety blocks, human review, provider errors, declines, timeouts, and bounded
escalation are represented as explicit statuses.

Key endpoints:

- `POST /api/donations` — create and process an offer
- `GET /api/donations` and `GET /api/donations/{id}` — inspect offers
- `POST /api/donations/{id}/clarify` — submit missing facts
- `POST /api/donations/{id}/safety-review` — resolve human safety review
- `POST /api/donations/{id}/approve` — record donor approval
- `POST /api/donations/{id}/ngo-response` — simulate `accept`, `decline`, or `timeout`
- `POST /api/donations/{id}/delivery-confirmation` — complete a delivery
- `POST /api/donations/{id}/retry` — retry an `ERROR` workflow
- `GET /api/donations/{id}/workflow` — view the workflow projection
- `GET /api/donations/{id}/agent-executions` — inspect agent/tool execution records
- `GET /api/donations/{id}/audit` — inspect the immutable audit trail
- `GET /api/ngos`, `/api/health`, `/api/metrics` — inspect system state
- `POST /api/demo/scenario/{scenario}` — run a synthetic Judge Lab scenario

Supported demo scenarios include `complete`, `missing`, `safety`, `rejection`,
`escalation`, `demand_pasta`, `demand_incompatible`, and
`demand_quantity_mismatch`.

## Tests and diagnostics

The default suite uses an injected mock client and does not call a live provider:

```powershell
.\.venv\Scripts\pytest
```

Run a live provider smoke test only when explicitly needed:

```powershell
$env:ANNAOPS_GEMINI_API_KEY="your_key"
.\.venv\Scripts\pytest -m live_llm
```

For a complete synthetic API workflow:

```powershell
.\.venv\Scripts\python.exe scripts\run_e2e_demo.py --scenario annaops_e2e_success
```

For a provider connectivity check:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_groq.py
```

Never commit `.env` or API keys. SQLite data is written to
`ANNAOPS_DATABASE_PATH` (default: `data/annaops.db`).

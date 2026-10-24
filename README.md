# AnnaOps: Surplus-Food Donation Coordination with OpenAI-Compatible LLM Providers

AnnaOps is a local-first, agentic surplus-food donation coordination system. It transforms raw donor messages into structured, validated food donation offers, enforces authoritative deterministic food safety rules, dynamically reasons over eligible receiver organizations, requests donor approval, simulates NGO responses, replans failed attempts, and maintains an immutable audit trail.

> **Important:** Every NGO, geographic coordinate, dispatch notification, and delivery confirmation in this repository is synthetic. AnnaOps is a safe simulation system for agentic workflow orchestration and evaluation.

---

## Architecture & Configured LLM Integration

AnnaOps uses the configured OpenAI-compatible provider for all designated agent operations. Groq is selected when `GROQ_API_KEY` is present; otherwise Gemini remains supported. Silent fallback to deterministic parsing or predetermined agent decisions is strictly prohibited.

### Canonical Configuration

| Setting | Canonical Value / Variable | Purpose |
| :--- | :--- | :--- |
| **Provider** | Groq or Google Gemini (OpenAI-compatible) | Generative intelligence provider |
| **Groq model** | `openai/gpt-oss-20b` (`GROQ_MODEL`) | Default cost-efficient agentic model |
| **Groq API key** | `GROQ_API_KEY` | Required when Groq is selected |
| **Groq base URL** | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint root |
| **Gemini model** | `gemini-3.6-flash` (`ANNAOPS_GEMINI_MODEL`) | Gemini fallback model |
| **Timeout** | `ANNAOPS_LLM_TIMEOUT_SECONDS` (default: `20`) | Strictly bounded per-call model timeout |
| **Max Iterations** | `ANNAOPS_AGENT_MAX_ITERATIONS` (default: `3`) | Bounded tool-calling execution ceiling |

Conflicting defaults (such as `gemini-2.5-flash` or `gemini-3.8-flash`) have been removed. The application strictly validates this configuration on startup.

---

## Agent Operations Powered by the Configured LLM

1. **Intake Agent (`annaops/agents/intake.py`)**:
   - Uses the configured provider/model for structured extraction from natural language donor offers.
   - Enforces Pydantic schema validation (`ExtractedOffer`) preserving food items, quantities, units, preparation timestamps, storage conditions, and coordinates.
   - Detects missing facts and contradictory timestamps, formulating targeted clarification questions.
   - If extraction fails or the provider is unreachable, an explicit failure is raised; silent fallback to a deterministic regex parser is prohibited.

2. **Receiver Advisor Agent (`annaops/agents/receiver_advisor.py`)**:
   - Executes inside a bounded tool-use runtime (`BoundedToolRuntime`).
   - Inspects real registered tool outputs (`retrieve_receiver_profiles`, `run_eligibility_and_scoring`, `inspect_previous_attempts`).
   - The configured model analyzes trade-offs and recommends an eligible receiver.
   - The recommendation is strictly validated against the authoritative eligible candidate set. If the model proposes an ineligible receiver or an invalid action, the operation explicitly fails with `AgentProtocolError`.

3. **Workflow Coordinator (`annaops/agents/coordinator.py`)**:
   - The configured model inspects persisted workflow state via registered tool `read_persisted_workflow_state` and proposes the next workflow operation.
   - Proposes actions from a bounded set (`run_intake`, `run_safety`, `run_matching`, `durable_pause`).
   - Authoritative backend code enforces state transition gates, role permissions, and idempotency.

4. **Dispatch & Replanning Agent (`annaops/agents/dispatch.py`)**:
   - When an NGO declines or times out, the configured model evaluates previous attempts, reasons about candidate alternatives, and proposes retry or escalation.
   - Hard retry limits (`ANNAOPS_MAX_DISPATCH_ATTEMPTS = 2`) and attempted-receiver exclusions are strictly enforced deterministically.

5. **Authoritative Deterministic Controls**:
   - Food safety rules, temperature/time limits, database transactions, role authorization gates, and state persistence are strictly deterministic. The LLM can never override a safety block, manufacture missing facts, or make an ineligible receiver eligible.

---

## Safe, Recoverable Error Handling

When an LLM provider error occurs, AnnaOps guarantees safe, non-corrupting behavior:
- **Missing API key or invalid model:** Fails application startup with actionable error guidance.
- **Provider unreachable / Rate limit (429) / Timeout:** Raises `LLMRetryableError` (HTTP 503 `retryable_provider_error`).
- **Malformed structured response:** Raises `LLMResponseError` (HTTP 502 `invalid_model_output`).
- **Invalid agent action / Ineligible receiver proposal:** Raises `AgentProtocolError` (HTTP 502 `agent_protocol_failure`).
- **Iteration limit exhausted:** Raises `AgentExecutionLimitError` (HTTP 502).

In all failure scenarios:
- The donation transitions safely to `ERROR` status (`WORKFLOW_ERROR`).
- An agent execution record is persisted with status `failed`, capturing the configured provider/model, error category, and diagnostics.
- The UI exposes a **Retry Processing** action that triggers `POST /api/donations/{id}/retry` once the provider recovers.

---

## Agent Command Centre (UI)

The UI explicitly visualizes the agentic workflow and distinguishes:
- **LLM Execution Succeeded** (Green badge, configured model identifier).
- **LLM Execution Failed** (Red badge with error category and diagnostic details).
- **Retryable Provider Failure** (Warning badge with retry guidance and a 1-click Retry button).
- **Awaiting Human Input** (Purple gate indicator for donor approvals or coordinators).
- **Deterministic Rule Enforcement** (Blue badge for safety, eligibility, and state transitions).

---

## Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.11+
- Google AI Studio Gemini API Key: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 2. Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Configure environment variables
Copy-Item .env.example .env
```

Edit `.env` to supply your API key:
```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1
ANNAOPS_LLM_BUDGET_PER_DONATION=5
ANNAOPS_LLM_BUDGET_EXTENSION=2
ANNAOPS_LLM_MAX_RETRIES=2
ANNAOPS_LLM_CONCURRENCY_LIMIT=2
```

### Groq diagnostic

Run one minimal live request using the same environment and configuration as the
application:

```powershell
python scripts/diagnose_groq.py
```

The command never prints the API key and exits non-zero when authentication,
permission/model access, rate limiting, endpoint, or transport errors occur.
A Groq HTTP 403 with error code `1010` indicates an organization or project
model-permission restriction; check the model permissions in Groq Console rather
than changing providers silently.

Quota controls reserve every actual Gemini request, including tool rounds and retries, in SQLite before the provider call. The default budget is five requests per donation, survives restarts, and cannot be reset by a repeated UI request. A coordinator can explicitly extend a donation's budget through `POST /api/donations/{id}/llm-budget-extension` using a coordinator actor; the extension is audited. Transport retries are bounded, use exponential backoff with jitter, honor `Retry-After`, and are globally concurrency-limited.

### Measured request reduction

The offline complete-workflow test uses the injected test client only to count model rounds (it does not contact Gemini). The previous graph re-entered the LLM coordinator after intake, safety, and matching: 8 coordinator rounds (two tool/final rounds at each of four coordinator nodes) plus 1 intake round and 2 receiver-advisor rounds = **11 model rounds**. The optimized graph runs the coordinator once and routes deterministic stage transitions directly: 2 coordinator rounds + 1 intake round + 2 receiver-advisor rounds = **5 model rounds**, a measured reduction of 6 rounds (54.5%). Live token counts remain unavailable unless the provider includes a `usage` object; the API reports that state explicitly rather than estimating it.

### 3. Run Application
```powershell
# Start Uvicorn dev server
.\.venv\Scripts\python.exe -m uvicorn annaops.main:app --env-file .env --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. API Swagger docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Run one complete live synthetic workflow

With the application running and the configured provider reachable, open a second PowerShell window and run:

```powershell
.\.venv\Scripts\python.exe scripts\run_e2e_demo.py --scenario annaops_e2e_success
```

The runner verifies redacted provider readiness and seeded synthetic demand, submits a fresh donation through the public API, checks the persisted intake/safety/matching evidence, performs explicit demo donor approval, simulates an eligible NGO acceptance, confirms synthetic delivery, and independently rereads the final donation, workflow, audit trail, and NGO capacity. It prints `PASS`, `FAIL`, or `NOT RUN` for all fourteen required stages and exits non-zero on any failure. It never prints the API key.

---

## Deployment Configuration

When deploying AnnaOps to hosted environments (e.g., Cloud Run, Render, Railway, AWS ECS):
1. **Never commit `.env` to version control.**
2. Configure environment variables in the hosting dashboard:
   - `ANNAOPS_GEMINI_API_KEY`: Your production Google AI Studio key.
   - `ANNAOPS_GEMINI_MODEL`: `gemini-3.6-flash` when Gemini is selected.
   - `ANNAOPS_GEMINI_BASE_URL`: `https://generativelanguage.googleapis.com/v1beta/openai`.
   - `ANNAOPS_DATABASE_PATH`: Path to persistent SQLite storage (e.g. `/data/annaops.db`).
   - `ANNAOPS_TIMEZONE`: e.g. `Asia/Kolkata` or your operational timezone.
3. If `ANNAOPS_GEMINI_API_KEY` is omitted, the production server will refuse to start and fail with a clear configuration error.

---

## Running Automated Tests

AnnaOps features a comprehensive test suite covering mandatory Gemini configuration, error handling, safety, and agent workflows.

### 1. Default Offline Test Suite (Using Injected Mock Client)
The automated test suite runs completely offline with mocked HTTP responses and test-only client injection (`TestMockClient`). No live API credits or internet connectivity are required:
```powershell
.\.venv\Scripts\pytest
```
All 50 tests will execute and pass deterministically.

### 2. Opt-in Live provider smoke test
An explicitly opt-in live test verifies real network connectivity and structured output extraction against the configured provider endpoint:
```powershell
# Set your real API key in the shell
$env:ANNAOPS_GEMINI_API_KEY="AIzaSy..."

# Execute the live test
.\.venv\Scripts\pytest -m live_llm
```

---

## API Summary

- `POST /api/donations`: Create and process a donation offer with the configured LLM provider/model.
- `POST /api/donations/{id}/retry`: Authorised retry of a failed donation after provider recovery.
- `GET /api/donations/{id}/agent-executions`: Full audit trail of agent executions, tool calls, and model outputs.
- `POST /api/donations/{id}/llm-budget-extension`: Audited coordinator-only recovery budget extension.
- `GET /api/donations/{id}/workflow`: Current 10-stage workflow projection and timing.
- `POST /api/donations/{id}/approve`: Donor approval gate.
- `POST /api/donations/{id}/ngo-response`: Simulated receiver response (`accept` / `decline` / `timeout`).
- `POST /api/donations/{id}/delivery-confirmation`: Final coordinator delivery confirmation.
- `GET /api/ngos`: List synthetic NGO profiles, active item demands, remaining demand, and capacity.
- `POST /api/demo/scenarios/{scenario}`: Run interactive Judge Lab demo scenarios (`complete`, `demand_pasta`, `demand_incompatible`, `demand_quantity_mismatch`, `missing`, `rejection`, `safety`, `escalation`).
- `GET /api/donations/{id}/record.csv`: Export synthetic donation record with disclaimer.

Synthetic NGO demand is seeded idempotently by `python -m annaops.seed`. Matching requires
the persisted demand item and unit to be compatible; kilograms are never treated as meal portions.
Acceptance atomically reserves the matching demand and receiver capacity in the local SQLite data layer.

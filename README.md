# AnnaOps: Surplus-Food Donation Coordination Powered by Gemini 3.6 Flash

AnnaOps is a local-first, agentic surplus-food donation coordination system. It transforms raw donor messages into structured, validated food donation offers, enforces authoritative deterministic food safety rules, dynamically reasons over eligible receiver organizations, requests donor approval, simulates NGO responses, replans failed attempts, and maintains an immutable audit trail.

> **Important:** Every NGO, geographic coordinate, dispatch notification, and delivery confirmation in this repository is synthetic. AnnaOps is a safe simulation system for agentic workflow orchestration and evaluation.

---

## Architecture & Mandatory Gemini 3.6 Flash Integration

In AnnaOps, **Gemini 3.6 Flash** (`gemini-3.6-flash`) is a **mandatory dependency** for all designated agent operations. Silent fallback to deterministic parsing or predetermined agent decisions is strictly prohibited.

### Canonical Configuration

| Setting | Canonical Value / Variable | Purpose |
| :--- | :--- | :--- |
| **Provider** | Google Gemini (OpenAI-compatible) | Generative intelligence provider |
| **Model ID** | `gemini-3.6-flash` (`ANNAOPS_GEMINI_MODEL`) | Strictly required canonical model identifier |
| **API Key** | `ANNAOPS_GEMINI_API_KEY` | Required API key from Google AI Studio |
| **Base URL** | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible endpoint root |
| **Timeout** | `ANNAOPS_LLM_TIMEOUT_SECONDS` (default: `20`) | Strictly bounded per-call model timeout |
| **Max Iterations** | `ANNAOPS_AGENT_MAX_ITERATIONS` (default: `3`) | Bounded tool-calling execution ceiling |

Conflicting defaults (such as `gemini-2.5-flash` or `gemini-3.8-flash`) have been removed. The application strictly validates this configuration on startup.

---

## Agent Operations Powered by Gemini 3.6 Flash

1. **Intake Agent (`annaops/agents/intake.py`)**:
   - Uses Gemini 3.6 Flash for structured extraction from natural language donor offers.
   - Enforces Pydantic schema validation (`ExtractedOffer`) preserving food items, quantities, units, preparation timestamps, storage conditions, and coordinates.
   - Detects missing facts and contradictory timestamps, formulating targeted clarification questions.
   - If extraction fails or the provider is unreachable, an explicit failure is raised; silent fallback to a deterministic regex parser is prohibited.

2. **Receiver Advisor Agent (`annaops/agents/receiver_advisor.py`)**:
   - Executes inside a bounded tool-use runtime (`BoundedToolRuntime`).
   - Inspects real registered tool outputs (`retrieve_receiver_profiles`, `run_eligibility_and_scoring`, `inspect_previous_attempts`).
   - Gemini 3.6 Flash analyzes trade-offs and recommends an eligible receiver.
   - The recommendation is strictly validated against the authoritative eligible candidate set. If the model proposes an ineligible receiver or an invalid action, the operation explicitly fails with `AgentProtocolError`.

3. **Workflow Coordinator (`annaops/agents/coordinator.py`)**:
   - Uses Gemini 3.6 Flash to inspect persisted workflow state via registered tool `read_persisted_workflow_state` and propose the next workflow operation.
   - Proposes actions from a bounded set (`run_intake`, `run_safety`, `run_matching`, `durable_pause`).
   - Authoritative backend code enforces state transition gates, role permissions, and idempotency.

4. **Dispatch & Replanning Agent (`annaops/agents/dispatch.py`)**:
   - When an NGO declines or times out, Gemini 3.6 Flash evaluates previous attempts, reasons about candidate alternatives, and proposes retry or escalation.
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
- An agent execution record is persisted with status `failed`, capturing the provider, model (`gemini-3.6-flash`), error category, and diagnostics.
- The UI exposes a **Retry Processing** action that triggers `POST /api/donations/{id}/retry` once the provider recovers.

---

## Agent Command Centre (UI)

The UI explicitly visualizes the agentic workflow and distinguishes:
- **LLM Execution Succeeded** (Green badge, model identifier `gemini-3.6-flash`).
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
ANNAOPS_GEMINI_API_KEY=AIzaSy...
ANNAOPS_GEMINI_MODEL=gemini-3.6-flash
ANNAOPS_GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
```

### 3. Run Application
```powershell
# Start Uvicorn dev server
.\.venv\Scripts\python.exe -m uvicorn annaops.main:app --env-file .env --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. API Swagger docs are at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Deployment Configuration

When deploying AnnaOps to hosted environments (e.g., Cloud Run, Render, Railway, AWS ECS):
1. **Never commit `.env` to version control.**
2. Configure environment variables in the hosting dashboard:
   - `ANNAOPS_GEMINI_API_KEY`: Your production Google AI Studio key.
   - `ANNAOPS_GEMINI_MODEL`: `gemini-3.6-flash`.
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

### 2. Opt-in Live Gemini 3.6 Flash Smoke Test
An explicitly opt-in live test verifies real network connectivity and structured output extraction against Google's Gemini 3.6 Flash endpoint:
```powershell
# Set your real API key in the shell
$env:ANNAOPS_GEMINI_API_KEY="AIzaSy..."

# Execute the live test
.\.venv\Scripts\pytest -m live_llm
```

---

## API Summary

- `POST /api/donations`: Create and process a donation offer with Gemini 3.6 Flash.
- `POST /api/donations/{id}/retry`: Authorised retry of a failed donation after provider recovery.
- `GET /api/donations/{id}/agent-executions`: Full audit trail of agent executions, tool calls, and model outputs.
- `GET /api/donations/{id}/workflow`: Current 10-stage workflow projection and timing.
- `POST /api/donations/{id}/approve`: Donor approval gate.
- `POST /api/donations/{id}/ngo-response`: Simulated receiver response (`accept` / `decline` / `timeout`).
- `POST /api/donations/{id}/delivery-confirmation`: Final coordinator delivery confirmation.
- `POST /api/demo/scenarios/{scenario}`: Run interactive Judge Lab demo scenarios (`complete`, `missing`, `rejection`, `safety`, `escalation`).
- `GET /api/donations/{id}/record.csv`: Export synthetic donation record with disclaimer.

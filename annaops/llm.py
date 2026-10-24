from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from annaops.config import CANONICAL_MODEL, Settings, settings


class LLMBudgetExceeded(RuntimeError):
    category = "budget_exhausted"


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for all LLM provider errors."""
    category: str = "unknown"


class LLMConfigurationError(LLMError):
    """Invalid credentials, unsupported model, or missing configuration."""
    category = "configuration_error"


class LLMRetryableError(LLMError):
    """Rate limit, temporary network failure, or timeout - safe to retry."""
    category = "retryable_provider_error"


class LLMResponseError(LLMError):
    """Malformed or invalid structured output from the model."""
    category = "structured_output_validation_failure"


# Backward-compatible alias
LLMUnavailable = LLMRetryableError


@dataclass(frozen=True)
class LLMInvocation:
    output: BaseModel
    provider: str
    model: str
    duration_ms: int
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class StructuredLLMClient:
    """OpenAI-compatible structured-output adapter for the configured provider."""

    _semaphores: dict[int, threading.BoundedSemaphore] = {}
    _semaphore_lock = threading.Lock()

    def __init__(self, config: Settings = settings, database: Any | None = None):
        self.config = config
        self.database = database
        with self._semaphore_lock:
            self._semaphore = self._semaphores.setdefault(
                config.llm_concurrency_limit,
                threading.BoundedSemaphore(config.llm_concurrency_limit),
            )

    @property
    def configured(self) -> bool:
        return bool(
            self.config.llm_api_key
            and self.config.llm_model
            and self.config.llm_provider in {"gemini", "groq"}
        )

    @property
    def provider(self) -> str:
        netloc = urlparse(self.config.llm_base_url).netloc
        if self.config.llm_provider:
            return self.config.llm_provider
        if "generativelanguage.googleapis.com" in netloc:
            return "gemini"
        return netloc or "unknown"

    def complete_structured(
        self,
        output_model: type[StructuredModel],
        *,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "none",
        donation_id: str | None = None,
        agent_type: str = "llm",
        workflow_stage: str = "unknown",
    ) -> LLMInvocation:
        if not self.config.llm_api_key:
            key_name = "GROQ_API_KEY" if self.config.llm_provider == "groq" else "ANNAOPS_GEMINI_API_KEY"
            raise LLMConfigurationError(
                f"{key_name} is not configured for provider '{self.config.llm_provider}'."
            )
        if not self.config.llm_model:
            raise LLMConfigurationError(
                f"No model is configured for provider '{self.config.llm_provider}'."
            )

        schema = output_model.model_json_schema()
        if tools:
            prompt = (
                f"{system_prompt}\nUse only the registered native tools supplied in this request. "
                "Do not invent any unregistered tool. After tool results are provided, "
                "return one object with all of these fields: current_task, observed_facts, "
                "missing_or_ambiguous, proposed_action, action_arguments, tool_calls, evidence, "
                "expected_outcome, human_input_required, and task_complete. "
                "Use an empty array or object for fields with no values."
            )
        else:
            prompt = (
                f"{system_prompt}\nReturn only one JSON object matching this JSON Schema. "
                "Do not include markdown or hidden reasoning.\n"
                f"Schema: {json.dumps(schema, separators=(',', ':'))}"
            )
        body = {
            "model": self.config.llm_model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": user_payload if isinstance(user_payload, str) else json.dumps(user_payload, default=str),
                },
            ],
        }
        if self.config.llm_provider == "groq" and self.config.llm_model == "openai/gpt-oss-20b":
            body["reasoning_effort"] = "low"
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__.lower(),
                    # Groq accepts these application schemas in best-effort mode;
                    # several contain intentionally open-ended dictionary fields.
                    "strict": self.config.llm_provider != "groq",
                    "schema": schema,
                },
            }
        request = urllib.request.Request(
            f"{self.config.llm_base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.config.llm_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "annaops/1.0",
            },
            method="POST",
        )
        max_attempts = self.config.llm_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            request_id = None
            if donation_id and self.database:
                request_id = self.database.reserve_llm_request(
                    donation_id, agent_type, workflow_stage, self.config.llm_budget_per_donation,
                    self.provider, self.config.llm_model,
                )
            try:
                with self._semaphore:
                    with urllib.request.urlopen(request, timeout=self.config.llm_timeout_seconds) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                message = payload["choices"][0]["message"]
                native_tool_calls = message.get("tool_calls") or []
                content = message.get("content")
                if native_tool_calls and tools and output_model.__name__ == "AgentDecision":
                    calls = []
                    for call in native_tool_calls:
                        function = call.get("function") or {}
                        try:
                            arguments = json.loads(function.get("arguments") or "{}")
                        except json.JSONDecodeError as exc:
                            raise LLMResponseError("Provider returned invalid tool arguments.") from exc
                        calls.append({
                            "name": function.get("name", ""),
                            "arguments": arguments,
                            "purpose": "Provider-selected registered tool",
                        })
                    output = output_model.model_validate({
                        "current_task": "Execute the provider-selected registered tools",
                        "observed_facts": [],
                        "missing_or_ambiguous": [],
                        "proposed_action": "use_tools",
                        "action_arguments": {},
                        "tool_calls": calls,
                        "evidence": ["Provider returned a native tool call."],
                        "expected_outcome": "Return validated tool results to the model.",
                        "human_input_required": False,
                        "task_complete": False,
                    })
                else:
                    output = output_model.model_validate_json(content or "")
                duration_ms = round((time.perf_counter() - started) * 1000)
                usage = payload.get("usage") or {}
                if request_id:
                    self.database.complete_llm_request(
                        request_id, outcome="succeeded", retry_count=attempt - 1,
                        input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                        latency_ms=duration_ms,
                    )
                return LLMInvocation(
                    output=output,
                    provider=self.provider,
                    model=self.config.llm_model,
                    duration_ms=duration_ms,
                    response_id=payload.get("id"),
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
            except urllib.error.HTTPError as exc:
                err_text = ""
                try:
                    err_text = exc.read().decode("utf-8")
                except Exception:
                    pass
                if exc.code == 401 or "API key" in err_text or "API_KEY" in err_text:
                    key_name = "GROQ_API_KEY" if self.config.llm_provider == "groq" else "ANNAOPS_GEMINI_API_KEY"
                    raise LLMConfigurationError(f"Authentication failed. Verify that {key_name} is valid.") from exc
                if exc.code == 403:
                    access_hint = ""
                    if "1010" in err_text or "model_permission" in err_text.lower():
                        access_hint = (
                            " Groq code 1010 indicates that organization or project model permissions "
                            "may block this model; check the Groq Console model permissions."
                        )
                    raise LLMConfigurationError(
                        f"Permission denied. The configured key does not have access to "
                        f"'{self.config.llm_model}'.{access_hint}"
                    ) from exc
                if exc.code == 404 or "not found" in err_text.lower():
                    raise LLMConfigurationError(
                        f"Model '{self.config.llm_model}' was not found by the provider. "
                        "Verify that the model identifier is correct and available to your account."
                    ) from exc
                retryable = exc.code in (408, 429, 500, 502, 503, 504)
                if request_id:
                    self.database.complete_llm_request(
                        request_id, outcome="failed" if attempt == max_attempts or not retryable else "retrying",
                        retry_count=attempt - 1, latency_ms=round((time.perf_counter() - started) * 1000),
                        error_category="rate_limit" if exc.code == 429 else "provider_error",
                    )
                if retryable and attempt < max_attempts:
                    retry_after = self._retry_after(exc)
                    time.sleep(min(30.0, retry_after if retry_after is not None else (0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25))))
                    continue
                if exc.code == 429:
                    raise LLMRetryableError(
                        f"Rate limit exceeded for '{self.config.llm_model}'. Safe to retry."
                    ) from exc
                if not retryable:
                    raise LLMConfigurationError(
                        f"Provider rejected the request with HTTP {exc.code}: {err_text[:200]}"
                    ) from exc
                raise LLMRetryableError(f"Provider returned HTTP {exc.code}: {err_text[:200]}") from exc
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                if request_id:
                    self.database.complete_llm_request(
                        request_id, outcome="failed" if attempt == max_attempts else "retrying",
                        retry_count=attempt - 1, latency_ms=round((time.perf_counter() - started) * 1000),
                        error_category="transport_error",
                    )
                if attempt < max_attempts:
                    time.sleep(min(30.0, 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25)))
                    continue
                raise LLMRetryableError(f"Provider was unreachable: {type(exc).__name__}") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                if request_id:
                    self.database.complete_llm_request(
                        request_id, outcome="failed", retry_count=attempt - 1,
                        latency_ms=round((time.perf_counter() - started) * 1000),
                        error_category="structured_output_validation_failure",
                    )
                raise LLMResponseError(
                    f"Provider returned an invalid structured response: {type(exc).__name__}: {str(exc)[:300]}"
                ) from exc

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> float | None:
        value = exc.headers.get("Retry-After") if exc.headers else None
        try:
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None


class TestMockClient:
    __test__ = False
    """Deterministic mock client used strictly in testing mode.

    Exposes provider='test_mock' and canonical model='gemini-3.6-flash'.
    A test mock is never mistaken for a real provider call.
    """

    configured = True
    provider = "test_mock"
    model = CANONICAL_MODEL

    def __init__(self, config: Settings | None = None):
        self.config = config or Settings.for_testing()
        self.calls: list[dict[str, Any]] = []

    def complete_structured(
        self,
        output_model: type[StructuredModel],
        *,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "none",
    ) -> LLMInvocation:
        self.calls.append({
            "output_model": output_model,
            "system_prompt": system_prompt,
            "user_payload": user_payload,
        })

        name = getattr(output_model, "__name__", "")
        if name == "ExtractedOffer":
            output = self._extract_offer(user_payload)
        elif name == "AgentDecision":
            output = self._agent_decision(system_prompt, user_payload)
        else:
            output = output_model()

        return LLMInvocation(
            output=output,
            provider=self.provider,
            model=self.model,
            duration_ms=3,
            response_id="mock-resp-id",
        )

    def _extract_offer(self, user_payload: dict[str, Any] | str) -> Any:
        from annaops.schemas import ExtractedOffer, FoodItem

        message = user_payload.get("original_donor_message", "") if isinstance(user_payload, dict) else str(user_payload)
        now = datetime.now(self.config.timezone).replace(microsecond=0)

        # Contradictions check
        contradictions: list[str] = []
        prep_iso = re.search(r"prepared at\s+([^\s,;]+)", message, re.IGNORECASE)
        dead_iso = re.search(r"pickup by\s+([^\s,;]+)", message, re.IGNORECASE)
        prepared_at: datetime | None = None
        pickup_deadline: datetime | None = None

        if prep_iso:
            try:
                prepared_at = datetime.fromisoformat(prep_iso.group(1).replace("Z", "+00:00"))
            except Exception:
                pass
        if dead_iso:
            try:
                pickup_deadline = datetime.fromisoformat(dead_iso.group(1).replace("Z", "+00:00"))
            except Exception:
                pass

        if prepared_at and pickup_deadline and pickup_deadline <= prepared_at:
            contradictions.append("Pickup deadline must be after preparation time.")

        # Food item extraction
        units = r"portions?|meals?|servings?|plates?|packs?|kgs?|kilograms?|litres?|liters?"
        stop = rf"(?=\s+(?:and|plus)\s+\d+(?:\.\d+)?\s*(?:{units})\b|,?\s*(?:prepared at|cooked at|pickup by|collect by|kept hot|refrigerated|chilled|frozen|room temperature|from|coordinates?)\b|[.;]|$)"
        pattern = re.compile(
            rf"\b(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>{units})\b\s*(?:of\s+)?(?P<food>.+?){stop}",
            re.IGNORECASE,
        )

        items: list[FoodItem] = []
        for match in pattern.finditer(message):
            food_raw = match.group("food").strip(" ,")
            food_name = re.sub(r"^(?:of\s+)", "", food_raw, flags=re.IGNORECASE).strip().lower()
            if not food_name:
                continue
            unit_norm = match.group("unit").lower()
            if unit_norm in ("kgs", "kilogram", "kilograms"):
                unit_norm = "kg"
            elif unit_norm in ("liter", "liters", "litre"):
                unit_norm = "litres"
            elif unit_norm in ("portion", "portions"):
                unit_norm = "portions"

            is_non_veg = any(w in food_name for w in ["chicken", "meat", "fish", "egg", "non-veg"])
            cat = "non_vegetarian" if is_non_veg else ("vegetarian" if "veg" in food_name else "unknown")
            items.append(FoodItem(
                name=food_name,
                quantity=float(match.group("qty")),
                unit=unit_norm,
                category=cat,
            ))

        # Check storage
        storage: str | None = None
        for pat in (r"\b(kept hot)\b", r"\b(refrigerated)\b", r"\b(chilled)\b", r"\b(room temperature)\b"):
            sm = re.search(pat, message, re.IGNORECASE)
            if sm:
                storage = sm.group(1).lower()
                break

        # Check location
        location: str | None = None
        loc_match = re.search(r"\bfrom\s+(.+?)(?=,\s*coordinates|[.;]|$)", message, re.IGNORECASE)
        if loc_match:
            location = loc_match.group(1).strip()
        elif "Demo Campus Jaipur" in message:
            location = "Demo Campus Jaipur"
        elif "Demo Hall" in message:
            location = "Demo Hall"

        lat, lon = None, None
        coord_match = re.search(r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", message)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))

        # Check for unstated fields
        missing: list[str] = []
        if not items:
            missing.extend(["food_type", "quantity", "unit"])
        if not prepared_at:
            missing.append("prepared_at")
        if not pickup_deadline:
            missing.append("pickup_deadline")
        if not storage:
            missing.append("storage_conditions")
        if not location:
            missing.append("donor_location")

        questions = [f"What is the {field}?" for field in missing]
        if contradictions:
            questions.append("Please correct conflicting timestamps.")

        explicit = []
        if items:
            explicit.extend(["food_items", "food_type", "quantity", "unit"])
        if prepared_at:
            explicit.append("prepared_at")
        if pickup_deadline:
            explicit.append("pickup_deadline")
        if storage:
            explicit.append("storage_conditions")
        if location:
            explicit.append("donor_location")
        if lat is not None and lon is not None:
            explicit.extend(["donor_latitude", "donor_longitude"])

        total_qty = sum(item.quantity for item in items) if len({item.unit for item in items}) == 1 else (items[0].quantity if len(items) == 1 else None)
        common_unit = items[0].unit if len({item.unit for item in items}) == 1 else None
        food_type_str = ", ".join(item.name for item in items) if items else ("vegetarian rice" if "vegetarian rice" in message.lower() else "surplus food")
        overall_cat = "non_vegetarian" if any(item.category == "non_vegetarian" for item in items) else ("vegetarian" if (items or "veg" in message.lower()) else "unknown")

        return ExtractedOffer(
            food_type=food_type_str,
            food_items=items,
            food_category=overall_cat,
            quantity=total_qty,
            unit=common_unit,
            prepared_at=prepared_at,
            pickup_deadline=pickup_deadline,
            storage_conditions=storage,
            donor_location=location,
            donor_latitude=lat,
            donor_longitude=lon,
            explicit_fields=explicit,
            inferred_fields=[],
            missing_fields=missing,
            contradictions=contradictions,
            clarification_questions=questions,
            extraction_source="configured_llm",
        )

    def _agent_decision(self, system_prompt: str, user_payload: dict[str, Any]) -> Any:
        from annaops.schemas import AgentDecision, AgentToolCall

        tool_results = user_payload.get("tool_results_so_far", [])

        if "coordinator" in system_prompt.lower():
            if not tool_results:
                return AgentDecision(
                    current_task="Inspect persisted workflow status before selecting next operation",
                    observed_facts=[],
                    missing_or_ambiguous=[],
                    proposed_action="use_tools",
                    action_arguments={},
                    tool_calls=[AgentToolCall(
                        name="read_persisted_workflow_state",
                        arguments={},
                        purpose="Read persisted workflow state",
                    )],
                    evidence=["State inspection required"],
                    expected_outcome="Read state",
                    human_input_required=False,
                    task_complete=False,
                )

            last_result = tool_results[-1].get("result", {})
            status = last_result.get("status", "INTAKE_PROCESSING")
            op_map = {
                "INTAKE_PROCESSING": "run_intake",
                "PENDING_SAFETY_REVIEW": "run_safety",
                "MATCHING": "run_matching",
            }
            operation = op_map.get(status, "durable_pause")
            return AgentDecision(
                current_task=f"Propose next operation for status {status}",
                observed_facts=[f"Current persisted status is {status}"],
                missing_or_ambiguous=[],
                proposed_action=operation,
                action_arguments={},
                tool_calls=[],
                evidence=[f"Status {status} maps to operation {operation}"],
                expected_outcome=f"Advance to {operation}",
                human_input_required=operation == "durable_pause",
                task_complete=True,
            )

        # Receiver advisor
        if not tool_results:
            return AgentDecision(
                current_task="Run authoritative eligibility and scoring before recommending a receiver",
                observed_facts=[],
                missing_or_ambiguous=[],
                proposed_action="use_tools",
                action_arguments={},
                tool_calls=[AgentToolCall(
                    name="run_eligibility_and_scoring",
                    arguments={},
                    purpose="Run authoritative eligibility and scoring",
                )],
                evidence=["Authoritative tool must run first"],
                expected_outcome="Obtain eligible scored candidates",
                human_input_required=False,
                task_complete=False,
            )

        last_result = tool_results[-1].get("result", {})
        candidates = last_result.get("candidates", [])
        if candidates:
            selected = candidates[0].get("ngo_id")
            return AgentDecision(
                current_task=f"Recommend top eligible receiver {selected}",
                observed_facts=[f"Found {len(candidates)} eligible receivers"],
                missing_or_ambiguous=[],
                proposed_action="recommend_receiver",
                action_arguments={"selected_ngo_id": selected},
                tool_calls=[],
                evidence=[f"Receiver {selected} is top ranked and eligible"],
                expected_outcome=f"Recommend {selected} for donor approval",
                human_input_required=True,
                task_complete=True,
            )

        return AgentDecision(
            current_task="Escalate because no eligible receiver exists",
            observed_facts=["Zero eligible receivers found"],
            missing_or_ambiguous=[],
            proposed_action="escalate",
            action_arguments={},
            tool_calls=[],
            evidence=["No eligible candidate exists"],
            expected_outcome="Escalate to human coordinator",
            human_input_required=True,
            task_complete=True,
        )

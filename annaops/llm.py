from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from annaops.config import Settings, settings


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMInvocation:
    output: BaseModel
    provider: str
    model: str
    duration_ms: int
    response_id: str | None = None


class StructuredLLMClient:
    """Small OpenAI-compatible structured-output adapter.

    It stores no credentials in results, validates every response with Pydantic, and deliberately
    exposes only concise outputs rather than internal reasoning.
    """

    def __init__(self, config: Settings = settings):
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.llm_api_key and self.config.llm_model)

    @property
    def provider(self) -> str:
        return urlparse(self.config.llm_base_url).netloc or "openai_compatible"

    def complete_structured(
        self,
        output_model: type[StructuredModel],
        *,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
    ) -> LLMInvocation:
        if not self.configured:
            raise LLMUnavailable("No LLM provider is configured")
        schema = output_model.model_json_schema()
        prompt = (
            f"{system_prompt}\nReturn only one JSON object matching this JSON Schema. "
            "Do not include markdown or hidden reasoning.\n"
            f"Schema: {json.dumps(schema, separators=(',', ':'))}"
        )
        body = {
            "model": self.config.llm_model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__.lower(),
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": user_payload if isinstance(user_payload, str) else json.dumps(user_payload, default=str),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.config.llm_base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.llm_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.config.llm_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            output = output_model.model_validate_json(content)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise LLMUnavailable("Configured model rate limit was reached") from exc
            raise LLMUnavailable(f"Configured model returned HTTP {exc.code}") from exc
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            raise LLMUnavailable(f"Configured model was unavailable: {type(exc).__name__}") from exc
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise LLMResponseError(f"Configured model returned an invalid structured response: {type(exc).__name__}") from exc
        duration_ms = round((time.perf_counter() - started) * 1000)
        return LLMInvocation(
            output=output,
            provider=self.provider,
            model=self.config.llm_model or "unknown",
            duration_ms=duration_ms,
            response_id=payload.get("id"),
        )

"""Run one redacted, minimal request against the configured Groq endpoint."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annaops.config import ConfigurationError, Settings


def _safe_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")[:300]
    except (OSError, UnicodeError):
        return ""


def _category(status: int | None, body: str) -> str:
    lowered = body.lower()
    if "<html" in lowered or "cloudflare" in lowered or "attention required" in lowered:
        return "intermediary_security_block"
    if status == 401:
        return "authentication"
    if status == 403 and ("1010" in lowered or "model_permission" in lowered):
        return "permission_or_model_access"
    if status == 403:
        return "permission"
    if status == 404:
        return "unsupported_model_or_endpoint"
    if status == 429:
        return "rate_limit_or_quota"
    if status is not None and status >= 500:
        return "provider_transient_error"
    return "provider_error"


def main() -> int:
    try:
        config = Settings.from_env()
        config.validate()
    except ConfigurationError as error:
        print(f"configured=false category=configuration error={error}")
        return 2

    if config.llm_provider != "groq":
        print(f"configured=false category=configuration provider={config.llm_provider!r}; Groq is not selected")
        return 2

    print(
        f"configured=true provider=groq model={config.llm_model} "
        f"endpoint={config.llm_base_url}/chat/completions api_key_present={bool(config.llm_api_key)}"
    )
    body = {
        "model": config.llm_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 4,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{config.llm_base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + (config.llm_api_key or ""),
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "annaops/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=min(config.llm_timeout_seconds, 10)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print(
            f"success=true status=200 content_type={response.headers.get('Content-Type')} "
            f"request_id={response.headers.get('x-request-id', 'unavailable')} "
            f"cf_ray={response.headers.get('cf-ray', 'unavailable')} "
            f"response_id={payload.get('id', 'unavailable')}"
        )
        return 0
    except urllib.error.HTTPError as error:
        safe_body = _safe_error_body(error)
        print(
            f"success=false status={error.code} content_type={error.headers.get('Content-Type')} "
            f"request_id={error.headers.get('x-request-id', 'unavailable')} "
            f"cf_ray={error.headers.get('cf-ray', 'unavailable')} "
            f"server={error.headers.get('server', 'unavailable')} "
            f"category={_category(error.code, safe_body)} "
            f"model={config.llm_model} detail={safe_body or 'no provider detail'}"
        )
        return 1
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        print(
            f"success=false status=unavailable category=transport_or_response_error "
            f"model={config.llm_model} detail={type(error).__name__}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

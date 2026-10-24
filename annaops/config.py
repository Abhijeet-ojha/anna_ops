from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


CANONICAL_MODEL = "gemini-3.6-flash"
CANONICAL_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class ConfigurationError(RuntimeError):
    """Raised when mandatory Gemini configuration is missing or invalid."""
    category = "configuration_error"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().strip('"').strip("'").lower() in {"1", "true", "yes", "on"}


def _clean_env(key: str, default: str = "") -> str:
    raw = os.getenv(key, default) or ""
    return raw.strip().strip('"').strip("'")


@dataclass(frozen=True)
class Settings:
    database_path: Path
    timezone_name: str
    demo_mode: bool
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str | None
    routing_base_url: str | None
    max_dispatch_attempts: int
    escalation_window_minutes: int
    safety_rule_version: str = "demo-safety-v1"
    matching_rule_version: str = "demo-match-v1"
    llm_timeout_seconds: int = 20
    agent_max_iterations: int = 3
    testing_mode: bool = False

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    def validate(self) -> None:
        """Validate mandatory Gemini configuration at startup.

        Raises ConfigurationError with an actionable message when required
        configuration is missing or invalid. Does not make a network call.
        """
        if self.testing_mode:
            return
        errors: list[str] = []
        if not self.llm_api_key or not self.llm_api_key.strip():
            errors.append(
                "ANNAOPS_GEMINI_API_KEY is required. "
                "Create an API key at https://aistudio.google.com/apikey "
                "and set the environment variable."
            )
        if not self.llm_model or not self.llm_model.strip():
            errors.append(
                f"ANNAOPS_GEMINI_MODEL must be set. "
                f"The canonical model for this project is '{CANONICAL_MODEL}'."
            )
        elif self.llm_model.strip() != CANONICAL_MODEL:
            errors.append(
                f"ANNAOPS_GEMINI_MODEL is set to '{self.llm_model}' but this project requires "
                f"'{CANONICAL_MODEL}'. Update the environment variable or remove it to use the canonical default."
            )
        if not self.llm_base_url or not self.llm_base_url.strip():
            errors.append(
                f"LLM base URL is empty. The canonical URL is '{CANONICAL_BASE_URL}'."
            )
        else:
            parsed = urlparse(self.llm_base_url.strip())
            if not parsed.scheme or not parsed.netloc:
                errors.append(
                    f"LLM base URL '{self.llm_base_url}' is not a valid URL. "
                    f"The canonical URL is '{CANONICAL_BASE_URL}'."
                )
        if errors:
            raise ConfigurationError(
                "AnnaOps startup failed due to missing or invalid Gemini configuration:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )

    @classmethod
    def from_env(cls) -> "Settings":
        key = _clean_env("ANNAOPS_GEMINI_API_KEY")
        model = _clean_env("ANNAOPS_GEMINI_MODEL", CANONICAL_MODEL)
        base_url = _clean_env("ANNAOPS_GEMINI_BASE_URL", CANONICAL_BASE_URL).rstrip("/")
        routing_url = _clean_env("ANNAOPS_ROUTING_BASE_URL").rstrip("/")

        return cls(
            database_path=Path(_clean_env("ANNAOPS_DATABASE_PATH", "data/annaops.db")),
            timezone_name=_clean_env("ANNAOPS_TIMEZONE", "Asia/Kolkata"),
            demo_mode=_as_bool(os.getenv("ANNAOPS_DEMO_MODE"), True),
            llm_api_key=key or None,
            llm_base_url=base_url or CANONICAL_BASE_URL,
            llm_model=model or CANONICAL_MODEL,
            routing_base_url=routing_url or None,
            max_dispatch_attempts=max(1, int(_clean_env("ANNAOPS_MAX_DISPATCH_ATTEMPTS", "2"))),
            escalation_window_minutes=max(1, int(_clean_env("ANNAOPS_ESCALATION_WINDOW_MINUTES", "30"))),
            llm_timeout_seconds=max(3, int(_clean_env("ANNAOPS_LLM_TIMEOUT_SECONDS", "20"))),
            agent_max_iterations=max(1, min(5, int(_clean_env("ANNAOPS_AGENT_MAX_ITERATIONS", "3")))),
        )

    @classmethod
    def for_testing(cls, **overrides) -> "Settings":
        """Create a Settings instance for tests with validation bypassed."""
        defaults = dict(
            database_path=Path("data/test.db"),
            timezone_name="Asia/Kolkata",
            demo_mode=True,
            llm_api_key="test-mock-key",
            llm_base_url=CANONICAL_BASE_URL,
            llm_model=CANONICAL_MODEL,
            routing_base_url=None,
            max_dispatch_attempts=2,
            escalation_window_minutes=30,
            testing_mode=True,
        )
        defaults.update(overrides)
        return cls(**defaults)


settings = Settings.from_env()

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    @classmethod
    def from_env(cls) -> "Settings":
        gemini_api_key = os.getenv("ANNAOPS_GEMINI_API_KEY") or None
        legacy_api_key = os.getenv("ANNAOPS_LLM_API_KEY") or None
        using_gemini = bool(gemini_api_key)
        return cls(
            database_path=Path(os.getenv("ANNAOPS_DATABASE_PATH", "data/annaops.db")),
            timezone_name=os.getenv("ANNAOPS_TIMEZONE", "Asia/Kolkata"),
            demo_mode=_as_bool(os.getenv("ANNAOPS_DEMO_MODE"), True),
            llm_api_key=gemini_api_key or legacy_api_key,
            llm_base_url=(
                (os.getenv("ANNAOPS_GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai")
                if using_gemini
                else (os.getenv("ANNAOPS_LLM_BASE_URL") or "https://api.openai.com/v1")
            ).rstrip("/"),
            llm_model=(
                (os.getenv("ANNAOPS_GEMINI_MODEL") or "gemini-3.8-flash")
                if using_gemini
                else (os.getenv("ANNAOPS_LLM_MODEL") or None)
            ),
            routing_base_url=(os.getenv("ANNAOPS_ROUTING_BASE_URL") or "").rstrip("/") or None,
            max_dispatch_attempts=max(1, int(os.getenv("ANNAOPS_MAX_DISPATCH_ATTEMPTS", "2"))),
            escalation_window_minutes=max(1, int(os.getenv("ANNAOPS_ESCALATION_WINDOW_MINUTES", "30"))),
            llm_timeout_seconds=max(3, int(os.getenv("ANNAOPS_LLM_TIMEOUT_SECONDS", "20"))),
            agent_max_iterations=max(1, min(5, int(os.getenv("ANNAOPS_AGENT_MAX_ITERATIONS", "3")))),
        )


settings = Settings.from_env()


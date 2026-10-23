from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from annaops.config import Settings
from annaops.main import create_app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_path=tmp_path / "annaops-test.db",
        timezone_name="Asia/Kolkata",
        demo_mode=True,
        llm_api_key=None,
        llm_base_url="https://example.invalid/v1",
        llm_model=None,
        routing_base_url=None,
        max_dispatch_attempts=2,
        escalation_window_minutes=30,
    )


@pytest.fixture
def client(test_settings: Settings):
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client


def complete_message(config: Settings, quantity: int = 40, storage: str = "kept hot") -> str:
    now = datetime.now(config.timezone).replace(microsecond=0)
    prepared = (now - timedelta(minutes=20)).isoformat()
    deadline = (now + timedelta(hours=3)).isoformat()
    return (
        f"{quantity} portions vegetarian biryani prepared at {prepared}, {storage}, "
        f"pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873"
    )


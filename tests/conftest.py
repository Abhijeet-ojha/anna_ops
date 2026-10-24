from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from annaops.config import CANONICAL_BASE_URL, CANONICAL_MODEL, Settings
from annaops.llm import TestMockClient as MockGeminiClient
from annaops.main import create_app


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings.for_testing(database_path=tmp_path / "annaops-test.db")


@pytest.fixture
def mock_llm_client(test_settings: Settings) -> MockGeminiClient:
    return MockGeminiClient(test_settings)


@pytest.fixture
def client(test_settings: Settings, mock_llm_client: MockGeminiClient):
    app = create_app(test_settings, llm_client=mock_llm_client)
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

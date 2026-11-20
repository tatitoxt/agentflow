from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterator

import pytest


@pytest.fixture
def redis_url() -> str:
    url = os.environ.get("AGENTFLOW_TEST_REDIS_URL")
    if not url:
        pytest.skip("set AGENTFLOW_TEST_REDIS_URL (e.g. redis://localhost:6379/0) to run")
    return url


@pytest.fixture
def redis_index(redis_url: str) -> Iterator[str]:
    redis = pytest.importorskip("redis")
    name = f"agentflow-test-{uuid.uuid4().hex[:8]}"
    yield name
    client = redis.Redis.from_url(redis_url)
    with contextlib.suppress(redis.ResponseError):
        client.ft(name).dropindex(delete_documents=True)
    client.close()

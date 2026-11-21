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
        if os.environ.get("AGENTFLOW_REQUIRE_REDIS") == "1":
            pytest.fail("AGENTFLOW_REQUIRE_REDIS=1 but AGENTFLOW_TEST_REDIS_URL is not set")
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
    # Plain-Redis stores use a <name>:ids set + <name>:doc:* hashes instead.
    with contextlib.suppress(redis.ResponseError):
        ids = [_id.decode() for _id in client.sscan_iter(f"{name}:ids")]
        if ids:
            client.delete(*[f"{name}:doc:{i}" for i in ids], f"{name}:ids")
    client.close()


@pytest.fixture
def pg_url() -> str:
    url = os.environ.get("AGENTFLOW_TEST_PG_URL")
    if not url:
        if os.environ.get("AGENTFLOW_REQUIRE_PG") == "1":
            pytest.fail("AGENTFLOW_REQUIRE_PG=1 but AGENTFLOW_TEST_PG_URL is not set")
        pytest.skip("set AGENTFLOW_TEST_PG_URL (e.g. postgresql://localhost:5432/db) to run")
    return url


@pytest.fixture
def pg_index(pg_url: str) -> Iterator[str]:
    psycopg = pytest.importorskip("psycopg")
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield name
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "agentflow_{name}"')

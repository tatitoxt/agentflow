"""Plain-Redis store: same behaviour as RedisVectorStore without RediSearch."""

from __future__ import annotations

import pytest

from agentflow.memory import HashingEmbedder, PlainRedisVectorStore, Record, VectorMemory

pytestmark = pytest.mark.redis

DOCS = [
    ("Refunds are processed within 5 business days of approval.", {"source": "billing.md"}),
    ("The public API is rate limited to 100 requests per minute.", {"source": "api.md"}),
    ("Single sign-on is available on the Enterprise plan.", {"source": "plans.md"}),
]


def test_plain_store_search_and_persistence(redis_url: str, redis_index: str) -> None:
    writer = VectorMemory(
        HashingEmbedder(),
        PlainRedisVectorStore(dim=256, url=redis_url, index_name=redis_index),
    )
    writer.add_many(DOCS)
    writer.add_many(DOCS)  # idempotent upsert

    reader = VectorMemory(
        HashingEmbedder(),
        PlainRedisVectorStore(dim=256, url=redis_url, index_name=redis_index),
    )
    assert len(reader) == 3
    hits = reader.search("API rate limit for refunds per minute", k=2)
    assert hits[0].metadata == {"source": "api.md"}
    assert hits[0].score > hits[1].score


def test_plain_store_clear_and_dimension_check(redis_url: str, redis_index: str) -> None:
    store = PlainRedisVectorStore(dim=4, url=redis_url, index_name=redis_index)
    store.upsert([Record(id="a", text="x", vector=[1.0, 0.0, 0.0, 0.0])])
    assert store.count() == 1
    with pytest.raises(ValueError, match="expected vector of dim 4"):
        store.upsert([Record(id="b", text="y", vector=[1.0])])
    store.clear()
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0, 0.0], k=3) == []

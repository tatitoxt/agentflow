from __future__ import annotations

import pytest

from agentflow import Agent, Tracer
from agentflow.memory import HashingEmbedder, Record, RedisVectorStore, VectorMemory
from agentflow.providers import FakeProvider

pytestmark = pytest.mark.redis

DOCS = [
    ("Refunds are processed within 5 business days of approval.", {"source": "billing.md"}),
    ("The public API is rate limited to 100 requests per minute.", {"source": "api.md"}),
    ("Single sign-on is available on the Enterprise plan.", {"source": "plans.md"}),
]


def test_knn_search_returns_closest_document(redis_url: str, redis_index: str) -> None:
    memory = VectorMemory(
        HashingEmbedder(), RedisVectorStore(dim=256, url=redis_url, index_name=redis_index)
    )
    memory.add_many(DOCS)
    assert len(memory) == 3
    hits = memory.search("API rate limit for refunds per minute", k=2)
    assert hits[0].metadata == {"source": "api.md"}
    assert hits[0].text == DOCS[1][0]
    assert hits[0].score > hits[1].score
    assert 0.0 < hits[0].score <= 1.0 + 1e-6


def test_documents_persist_across_store_instances(redis_url: str, redis_index: str) -> None:
    writer = VectorMemory(
        HashingEmbedder(), RedisVectorStore(dim=256, url=redis_url, index_name=redis_index)
    )
    writer.add_many(DOCS)
    writer.add_many(DOCS)  # idempotent upsert

    reader = VectorMemory(
        HashingEmbedder(), RedisVectorStore(dim=256, url=redis_url, index_name=redis_index)
    )
    assert len(reader) == 3
    assert reader.search("enterprise single sign-on", k=1)[0].metadata["source"] == "plans.md"


def test_clear_and_dimension_check(redis_url: str, redis_index: str) -> None:
    store = RedisVectorStore(dim=4, url=redis_url, index_name=redis_index)
    store.upsert([Record(id="a", text="x", vector=[1.0, 0.0, 0.0, 0.0])])
    assert store.count() == 1
    with pytest.raises(ValueError, match="expected vector of dim 4"):
        store.upsert([Record(id="b", text="y", vector=[1.0])])
    store.clear()
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0, 0.0], k=3) == []


def test_agent_retrieval_from_redis(redis_url: str, redis_index: str) -> None:
    memory = VectorMemory(
        HashingEmbedder(), RedisVectorStore(dim=256, url=redis_url, index_name=redis_index)
    )
    memory.add_many(DOCS)
    provider = FakeProvider([lambda messages, tools: messages[0].content.splitlines()[1]])
    tracer = Tracer()
    result = Agent("rag", provider, memory=memory, memory_k=1, tracer=tracer).run(
        "How long do refunds take?"
    )
    assert result.output == f"[1] (source: billing.md) {DOCS[0][0]}"
    assert [e.type for e in tracer.events][:2] == ["run_start", "memory_retrieval"]

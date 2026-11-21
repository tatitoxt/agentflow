"""PgVectorStore: persistent memory backed by Postgres + pgvector."""

from __future__ import annotations

import pytest

from agentflow.memory import HashingEmbedder, PgVectorStore, Record, VectorMemory

pytestmark = pytest.mark.pgvector

DOCS = [
    ("Refunds are processed within 5 business days of approval.", {"source": "billing.md"}),
    ("The public API is rate limited to 100 requests per minute.", {"source": "api.md"}),
    ("Single sign-on is available on the Enterprise plan.", {"source": "plans.md"}),
]


def test_pgvector_search_and_persistence(pg_url: str, pg_index: str) -> None:
    writer = VectorMemory(
        HashingEmbedder(), PgVectorStore(dim=256, url=pg_url, index_name=pg_index)
    )
    writer.add_many(DOCS)
    writer.add_many(DOCS)  # idempotent upsert

    reader = VectorMemory(
        HashingEmbedder(), PgVectorStore(dim=256, url=pg_url, index_name=pg_index)
    )
    assert len(reader) == 3
    hits = reader.search("API rate limit for refunds per minute", k=2)
    assert hits[0].metadata == {"source": "api.md"}
    assert hits[0].score > hits[1].score


def test_pgvector_clear_and_dimension_check(pg_url: str, pg_index: str) -> None:
    store = PgVectorStore(dim=4, url=pg_url, index_name=pg_index)
    store.upsert([Record(id="a", text="x", vector=[1.0, 0.0, 0.0, 0.0])])
    assert store.count() == 1
    with pytest.raises(ValueError, match="expected vector of dim 4"):
        store.upsert([Record(id="b", text="y", vector=[1.0])])
    with pytest.raises(ValueError, match="index_name must be a valid SQL identifier"):
        PgVectorStore(dim=4, url=pg_url, index_name='x"; DROP TABLE')
    store.clear()
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0, 0.0], k=3) == []

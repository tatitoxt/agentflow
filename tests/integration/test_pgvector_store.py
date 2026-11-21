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


def test_pgvector_ensure_index_and_configure_search(pg_url: str, pg_index: str) -> None:
    store = PgVectorStore(dim=4, url=pg_url, index_name=pg_index)
    store.upsert(
        [
            Record(id=str(i), text=f"doc {i}", vector=[float(i % 3), 0.0, 0.0, 1.0])
            for i in range(10)
        ]
    )
    name = store.ensure_index()  # hnsw default
    assert name == f"{store.table}_hnsw_idx"
    assert store.ensure_index() == name  # idempotent
    assert store.ensure_index("ivfflat", lists=5) == f"{store.table}_ivfflat_idx"
    with pytest.raises(ValueError, match="method must be"):
        store.ensure_index("brute")

    store.configure_search(ef_search=40)
    store.configure_search(probes=3)
    hits = store.search([1.0, 0.0, 0.0, 1.0], k=3)
    assert len(hits) == 3

    # the index exists and the planner can use it
    rows = store.conn.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = %s", (store.table,)
    ).fetchall()
    assert {f"{store.table}_hnsw_idx", f"{store.table}_ivfflat_idx"} <= {r[0] for r in rows}


def test_pgvector_auto_index_threshold(pg_url: str, pg_index: str) -> None:
    store = PgVectorStore(dim=4, url=pg_url, index_name=pg_index)
    # Below the threshold an exact scan stays in place.
    assert store.auto_index(threshold=10) is None
    assert store.index_exists() is None

    store.upsert(
        [Record(id=str(i), text=f"doc {i}", vector=[float(i), 0.0, 0.0, 1.0]) for i in range(12)]
    )
    name = store.auto_index(threshold=10)
    assert name == f"{store.table}_hnsw_idx"
    assert store.index_exists() == name
    # Second call is a no-op returning the existing index.
    assert store.auto_index(threshold=10_000) == name
    # Search still returns sane results through the index.
    hits = store.search([11.0, 0.0, 0.0, 1.0], k=1)
    assert hits[0][0].id == "11"

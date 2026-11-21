"""Vector stores: an in-memory store, a Redis Stack (RediSearch) store, and a plain-Redis store."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Record:
    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, records: Sequence[Record]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Record, float]]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    """Brute-force cosine search over a dict. Not persistent; ideal for tests and demos."""

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def upsert(self, records: Sequence[Record]) -> None:
        for record in records:
            self._records[record.id] = record

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Record, float]]:
        scored = [(r, cosine(vector, r.vector)) for r in self._records.values()]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[:k]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()


class RedisVectorStore:
    """Persistent store backed by Redis Stack: records are hashes, KNN via a RediSearch index.

    Requires the RediSearch module (``redis/redis-stack-server`` image or Redis 8+).
    Vectors are stored as FLOAT32 bytes; similarity is ``1 - cosine distance``.
    """

    def __init__(
        self,
        *,
        dim: int,
        url: str = "redis://localhost:6379/0",
        index_name: str = "agentflow",
        client: Any | None = None,
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("RedisVectorStore needs `pip install agentflow[redis]`") from exc
        self.dim = dim
        self.index_name = index_name
        self.prefix = f"{index_name}:doc:"
        self._redis = redis
        self.client = client or redis.Redis.from_url(url)
        self._ensure_index()

    def _ensure_index(self) -> None:
        from redis.commands.search.field import TextField, VectorField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        try:
            self.client.ft(self.index_name).info()
            return
        except self._redis.ResponseError:
            pass
        self.client.ft(self.index_name).create_index(
            [
                TextField("text"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": "COSINE"},
                ),
            ],
            definition=IndexDefinition(  # type: ignore[no-untyped-call]
                prefix=[self.prefix], index_type=IndexType.HASH
            ),
        )

    def _pack(self, vector: Sequence[float]) -> bytes:
        if len(vector) != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got {len(vector)}")
        return struct.pack(f"<{self.dim}f", *vector)

    def upsert(self, records: Sequence[Record]) -> None:
        pipe = self.client.pipeline(transaction=False)
        for r in records:
            pipe.hset(
                f"{self.prefix}{r.id}",
                mapping={
                    "record_id": r.id,
                    "text": r.text,
                    "metadata": json.dumps(r.metadata, sort_keys=True),
                    "embedding": self._pack(r.vector),
                },
            )
        pipe.execute()

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Record, float]]:
        from redis.commands.search.query import Query

        query = (
            Query(f"*=>[KNN {int(k)} @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("record_id", "text", "metadata", "distance")
            .paging(0, k)
            .dialect(2)
        )
        result = self.client.ft(self.index_name).search(
            query, query_params={"vec": self._pack(vector)}
        )
        hits: list[tuple[Record, float]] = []
        for doc in result.docs:
            record = Record(
                id=_text(doc.record_id),
                text=_text(doc.text),
                vector=[],
                metadata=json.loads(_text(doc.metadata) or "{}"),
            )
            hits.append((record, 1.0 - float(_text(doc.distance))))
        hits.sort(key=lambda item: (-item[1], item[0].id))
        return hits

    def count(self) -> int:
        info = self.client.ft(self.index_name).info()
        return int(info["num_docs"])

    def clear(self) -> None:
        """Drop the index and its documents."""
        self.client.ft(self.index_name).dropindex(delete_documents=True)
        self._ensure_index()


class PlainRedisVectorStore:
    """Persistent store for plain Redis — no RediSearch module required.

    Records are the same hashes as :class:`RedisVectorStore` (so both stores can
    share a prefix), with ids tracked in a ``<index>:ids`` set; search fetches
    each hash and computes cosine in Python. Fine for small memories; for
    thousands of records prefer the RediSearch-backed store.
    """

    def __init__(
        self,
        *,
        dim: int,
        url: str = "redis://localhost:6379/0",
        index_name: str = "agentflow",
        client: Any | None = None,
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "PlainRedisVectorStore needs `pip install agentflow[redis]`"
            ) from exc
        self.dim = dim
        self.index_name = index_name
        self.prefix = f"{index_name}:doc:"
        self._ids_key = f"{index_name}:ids"
        self.client: Any = client or redis.Redis.from_url(url)

    def _pack(self, vector: Sequence[float]) -> bytes:
        if len(vector) != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got {len(vector)}")
        return struct.pack(f"<{self.dim}f", *vector)

    def _unpack(self, blob: bytes) -> list[float]:
        return list(struct.unpack(f"<{self.dim}f", blob))

    def upsert(self, records: Sequence[Record]) -> None:
        pipe = self.client.pipeline(transaction=False)
        for r in records:
            pipe.hset(
                f"{self.prefix}{r.id}",
                mapping={
                    "record_id": r.id,
                    "text": r.text,
                    "metadata": json.dumps(r.metadata, sort_keys=True),
                    "embedding": self._pack(r.vector),
                },
            )
            pipe.sadd(self._ids_key, r.id)
        pipe.execute()

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Record, float]]:
        if len(vector) != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got {len(vector)}")
        hits: list[tuple[Record, float]] = []
        for raw_id in self.client.sscan_iter(self._ids_key):
            doc = self.client.hgetall(f"{self.prefix}{_text(raw_id)}")
            if not doc:
                continue
            record = Record(
                id=_text(doc.get(b"record_id", raw_id)),
                text=_text(doc.get(b"text", b"")),
                vector=[],
                metadata=json.loads(_text(doc.get(b"metadata", b"{}")) or "{}"),
            )
            blob = doc.get(b"embedding", b"")
            score = cosine(vector, self._unpack(blob)) if blob else 0.0
            hits.append((record, score))
        hits.sort(key=lambda item: (-item[1], item[0].id))
        return hits[:k]

    def count(self) -> int:
        return int(self.client.scard(self._ids_key))

    def clear(self) -> None:
        ids = [_text(raw) for raw in self.client.sscan_iter(self._ids_key)]
        if ids:
            self.client.delete(*[f"{self.prefix}{i}" for i in ids], self._ids_key)
        else:
            self.client.delete(self._ids_key)


class PgVectorStore:
    """Persistent store backed by Postgres + pgvector.

    One table per ``index_name`` (``agentflow_<index_name>``): id, text,
    metadata (jsonb) and a ``vector(dim)`` column. Search orders by cosine
    distance (``<=>``); scores are ``1 - distance``. Search is exact until
    ``ensure_index()`` builds an HNSW/IVFFlat index for large corpora, with
    recall tuned via ``configure_search()``.
    """

    def __init__(
        self,
        *,
        dim: int,
        url: str = "postgresql://localhost:5432/agentflow",
        index_name: str = "agentflow",
        client: Any | None = None,
    ) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("PgVectorStore needs `pip install agentflow[postgres]`") from exc
        if not index_name.isidentifier():
            raise ValueError(f"index_name must be a valid SQL identifier, got {index_name!r}")
        self.dim = dim
        self.index_name = index_name
        self.table = f"agentflow_{index_name}"
        self.conn: Any = client or psycopg.connect(url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{self.table}" ('
            "id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}',"
            f" embedding vector({int(dim)}) NOT NULL)"
        )

    def _pack(self, vector: Sequence[float]) -> str:
        if len(vector) != self.dim:
            raise ValueError(f"expected vector of dim {self.dim}, got {len(vector)}")
        return "[" + ",".join(repr(v) for v in vector) + "]"

    def upsert(self, records: Sequence[Record]) -> None:
        for r in records:
            self.conn.execute(
                f'INSERT INTO "{self.table}" (id, text, metadata, embedding)'
                " VALUES (%s,%s,%s,%s::vector)"
                " ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text,"
                " metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding",
                (r.id, r.text, json.dumps(r.metadata, sort_keys=True), self._pack(r.vector)),
            )

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Record, float]]:
        rows = self.conn.execute(
            f'SELECT id, text, metadata, embedding <=> %s::vector AS distance FROM "{self.table}"'
            " ORDER BY distance, id LIMIT %s",
            (self._pack(vector), int(k)),
        ).fetchall()
        return [
            (
                Record(id=row[0], text=row[1], vector=[], metadata=dict(row[2] or {})),
                1.0 - float(row[3]),
            )
            for row in rows
        ]

    def count(self) -> int:
        return int(self.conn.execute(f'SELECT count(*) FROM "{self.table}"').fetchone()[0])

    def clear(self) -> None:
        self.conn.execute(f'DELETE FROM "{self.table}"')

    def ensure_index(
        self,
        method: str = "hnsw",
        *,
        m: int = 16,
        ef_construction: int = 64,
        lists: int = 100,
    ) -> str:
        """Create an approximate index over ``embedding`` (cosine ops).

        ``method="hnsw"`` (pgvector >= 0.5) builds an HNSW graph index;
        ``method="ivfflat"`` builds an IVFFlat one — useful on older pgvector
        or when a small, fast ``lists`` scan is preferred. Idempotent
        (``CREATE INDEX IF NOT EXISTS``); returns the index name.
        """
        if method not in {"hnsw", "ivfflat"}:
            raise ValueError(f"method must be 'hnsw' or 'ivfflat', got {method!r}")
        name = f"{self.table}_{method}_idx"
        if method == "hnsw":
            options = f"m = {int(m)}, ef_construction = {int(ef_construction)}"
        else:
            options = f"lists = {int(lists)}"
        self.conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{name}" ON "{self.table}"'
            f" USING {method} (embedding vector_cosine_ops) WITH ({options})"
        )
        return name

    def configure_search(
        self, *, probes: int | None = None, ef_search: int | None = None
    ) -> None:
        """Tune ANN recall for this connection's queries.

        ``probes`` sets ``ivfflat.probes`` (IVFFlat index); ``ef_search`` sets
        ``hnsw.ef_search`` (HNSW index). Higher values = better recall, slower
        queries. Applies to the whole session until changed.
        """
        if probes is not None:
            self.conn.execute(f"SET ivfflat.probes = {int(probes)}")
        if ef_search is not None:
            self.conn.execute(f"SET hnsw.ef_search = {int(ef_search)}")


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

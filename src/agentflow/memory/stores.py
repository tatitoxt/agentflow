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
        self.client = client or redis.Redis.from_url(url)

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


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)

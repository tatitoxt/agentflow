"""VectorMemory: embed + store + retrieve text snippets."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embedders import Embedder
from .stores import Record, VectorStore


@dataclass
class MemoryHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def content_id(text: str, metadata: dict[str, Any] | None = None) -> str:
    """Stable id so re-adding the same snippet is an idempotent upsert."""
    source = str((metadata or {}).get("source", ""))
    return hashlib.sha1(f"{source}\x00{text}".encode()).hexdigest()[:16]


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Split on blank lines, packing paragraphs into chunks of at most ``max_chars``."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


class VectorMemory:
    """Pairs an :class:`Embedder` with a :class:`VectorStore`."""

    def __init__(self, embedder: Embedder, store: VectorStore, *, min_score: float = 0.0) -> None:
        self.embedder = embedder
        self.store = store
        self.min_score = min_score

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        return self.add_many([(text, metadata or {})])[0]

    def add_many(self, items: Iterable[tuple[str, dict[str, Any]]]) -> list[str]:
        pairs = list(items)
        vectors = self.embedder.embed([text for text, _ in pairs])
        records = [
            Record(id=content_id(text, meta), text=text, vector=vec, metadata=dict(meta))
            for (text, meta), vec in zip(pairs, vectors, strict=True)
        ]
        self.store.upsert(records)
        return [r.id for r in records]

    def add_directory(
        self, path: str | Path, pattern: str = "*.md", max_chars: int = 600
    ) -> list[str]:
        """Chunk and index every file matching ``pattern`` (metadata: source, chunk)."""
        items: list[tuple[str, dict[str, Any]]] = []
        for file in sorted(Path(path).glob(pattern)):
            for i, chunk in enumerate(chunk_text(file.read_text(encoding="utf-8"), max_chars)):
                items.append((chunk, {"source": file.name, "chunk": i}))
        return self.add_many(items) if items else []

    def search(self, query: str, k: int = 3) -> list[MemoryHit]:
        if k <= 0:
            return []
        vector = self.embedder.embed([query])[0]
        return [
            MemoryHit(id=r.id, text=r.text, score=score, metadata=r.metadata)
            for r, score in self.store.search(vector, k)
            if score > self.min_score
        ]

    def __len__(self) -> int:
        return self.store.count()


__all__: Sequence[str] = ("MemoryHit", "VectorMemory", "chunk_text", "content_id")

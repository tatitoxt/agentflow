"""Embedders turn text into fixed-size float vectors."""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

from ..errors import ConfigurationError, ProviderError

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it of on or our the this to "
    "what when where which who why with you your".split()
)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbedder:
    """Offline, deterministic embedder using the signed hashing trick.

    Unigrams and word bigrams are hashed with BLAKE2b (stable across processes, unlike
    ``hash()``) into ``dim`` buckets with log-scaled term frequency, then L2-normalized.
    It captures lexical overlap only — no semantics — which is enough for demos and tests.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def _features(self, text: str) -> Counter[str]:
        tokens = tokenize(text)
        features = Counter(tokens)
        features.update(f"{a} {b}" for a, b in itertools.pairwise(tokens))
        return features

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            weight = 1.0 + math.log(count)
            if " " in feature:
                weight *= 0.5
            vector[(value >> 1) % self.dim] += sign * weight
        return normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI embeddings over httpx (``POST /v1/embeddings``). Opt-in; needs an API key."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def from_env(cls, **kwargs: Any) -> OpenAIEmbedder:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("set OPENAI_API_KEY to use OpenAIEmbedder")
        return cls(api_key=api_key, **kwargs)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dim != 1536 or self.model != "text-embedding-3-small":
            payload["dimensions"] = self.dim
        try:
            response = self._client.post(
                f"{self.base_url}/embeddings", json=payload, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"openai embeddings: request failed: {exc}") from exc
        if response.status_code >= 400:
            raise ProviderError(
                f"openai embeddings: HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
            )
        items = sorted(response.json()["data"], key=lambda d: d["index"])
        vectors = [[float(x) for x in item["embedding"]] for item in items]
        if any(len(v) != self.dim for v in vectors):
            raise ProviderError(f"openai embeddings: expected dim {self.dim}")
        return vectors

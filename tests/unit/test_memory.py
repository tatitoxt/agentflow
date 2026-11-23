from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agentflow import Agent, Tracer
from agentflow.errors import ProviderError
from agentflow.memory import (
    HashingEmbedder,
    InMemoryVectorStore,
    OpenAIEmbedder,
    VectorMemory,
    chunk_text,
    content_id,
    cosine,
    tokenize,
)
from agentflow.providers import FakeProvider


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["Reset your password", "Reset your password"])
    assert a == b and len(a) == 64
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
    assert emb.embed([""])[0] == [0.0] * 64


def test_hashing_embedder_ranks_lexical_overlap() -> None:
    emb = HashingEmbedder()
    query = emb.embed_one("how do I reset my password")
    related = emb.embed_one("To reset a password open Settings and choose Reset password")
    unrelated = emb.embed_one("Invoices are emailed on the first business day of the month")
    assert cosine(query, related) > cosine(query, unrelated)


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What is the API rate-limit?") == ["api", "rate", "limit"]


def test_chunk_text_packs_paragraphs() -> None:
    text = "one\n\ntwo\n\n" + "x" * 30
    assert chunk_text(text, max_chars=10) == ["one\n\ntwo", "x" * 30]


def test_in_memory_store_search_and_upsert_idempotent() -> None:
    memory = VectorMemory(HashingEmbedder(), InMemoryVectorStore())
    first = memory.add("Refunds are processed within 5 business days", {"source": "billing.md"})
    memory.add("The API is rate limited to 100 requests per minute", {"source": "api.md"})
    again = memory.add("Refunds are processed within 5 business days", {"source": "billing.md"})
    assert (
        first
        == again
        == content_id("Refunds are processed within 5 business days", {"source": "billing.md"})
    )
    assert len(memory) == 2
    hits = memory.search("how long do refunds take", k=1)
    assert hits[0].metadata == {"source": "billing.md"}
    assert memory.search("anything", k=0) == []


def test_min_score_filters_unrelated_hits() -> None:
    memory = VectorMemory(HashingEmbedder(), InMemoryVectorStore(), min_score=0.2)
    memory.add("Refunds are processed within 5 business days")
    assert memory.search("kubernetes helm chart", k=3) == []


def test_add_directory_indexes_chunks_with_source(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nAlpha paragraph about refunds.")
    (tmp_path / "b.txt").write_text("ignored")
    memory = VectorMemory(HashingEmbedder(), InMemoryVectorStore())
    ids = memory.add_directory(tmp_path)
    assert len(ids) == 1
    assert memory.search("refunds")[0].metadata == {"source": "a.md", "chunk": 0}


def test_agent_injects_retrieved_context() -> None:
    memory = VectorMemory(HashingEmbedder(), InMemoryVectorStore())
    memory.add("The office wifi password rotates every Monday", {"source": "it.md"})
    provider = FakeProvider(["ok"])
    tracer = Tracer()
    Agent("rag", provider, memory=memory, memory_k=1, tracer=tracer).run("wifi password?")
    sent = provider.calls[0][0]
    assert sent[0].role == "system"
    assert sent[0].content.startswith("Relevant context retrieved from memory:")
    assert "(source: it.md) The office wifi password rotates" in sent[0].content
    retrieval = next(e for e in tracer.events if e.type == "memory_retrieval")
    assert retrieval.data["hits"][0]["metadata"] == {"source": "it.md"}


def test_openai_embedder_wire_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1, 0]}]
            },
        )

    emb = OpenAIEmbedder(
        api_key="sk",
        dim=2,
        model="text-embedding-3-small",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    body = json.loads(seen[0].content)
    assert body == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2}
    assert str(seen[0].url) == "https://api.openai.com/v1/embeddings"
    assert emb.embed([]) == []


def test_openai_embedder_http_error() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="no")))
    with pytest.raises(ProviderError, match="HTTP 401"):
        OpenAIEmbedder(api_key="sk", client=client).embed(["a"])

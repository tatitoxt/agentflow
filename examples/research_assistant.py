"""Research assistant over local Markdown docs (RAG).

Indexes ``examples/data/handbook`` into vector memory, injects the top matches into the
agent context and gives the agent a ``read_document`` tool to read a full source file.

Run offline (deterministic FakeProvider):

    agentflow run examples/research_assistant.py -i "How long do refunds take?"

Set ``AGENTFLOW_REDIS_URL`` to store vectors in Redis Stack instead of in memory, or
``AGENTFLOW_PROVIDER=openai|anthropic|groq`` (plus the API key) to use a real model.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from agentflow import Agent, Message, ToolSpec, tool
from agentflow.memory import (
    HashingEmbedder,
    InMemoryVectorStore,
    RedisVectorStore,
    VectorMemory,
    VectorStore,
    tokenize,
)
from agentflow.providers import FakeProvider, Provider

DOCS_DIR = Path(__file__).parent / "data" / "handbook"
DEFAULT_INPUT = "How long do refunds take?"
NOT_FOUND = "I could not find this in the handbook."

INSTRUCTIONS = """You answer questions about the Acme Notes handbook.
Use the retrieved context and the read_document tool. Answer in one or two sentences and
cite the source file in square brackets, e.g. [source: billing.md]. If the handbook does not
cover the question, say so."""


def build_memory() -> VectorMemory:
    embedder = HashingEmbedder(dim=256)
    redis_url = os.environ.get("AGENTFLOW_REDIS_URL")
    store: VectorStore = (
        RedisVectorStore(dim=embedder.dim, url=redis_url, index_name="agentflow-handbook")
        if redis_url
        else InMemoryVectorStore()
    )
    memory = VectorMemory(embedder, store, min_score=0.1)
    memory.add_directory(DOCS_DIR, max_chars=400)
    return memory


@tool
def read_document(source: str) -> str:
    """Return the full text of a handbook document.

    Args:
        source: File name as shown in the retrieved context, e.g. "billing.md".
    """
    path = (DOCS_DIR / source).resolve()
    if path.parent != DOCS_DIR.resolve() or not path.is_file():
        available = ", ".join(sorted(p.name for p in DOCS_DIR.glob("*.md")))
        raise FileNotFoundError(f"no document named {source!r}; available: {available}")
    return path.read_text(encoding="utf-8")


def build_agent(provider: Provider) -> Agent:
    return Agent(
        "researcher",
        provider,
        instructions=INSTRUCTIONS,
        tools=[read_document],
        memory=build_memory(),
        memory_k=2,
        max_steps=4,
    )


# --- Deterministic policy used by the FakeProvider ------------------------------------------


def _question(messages: list[Message]) -> str:
    return next(m.content for m in reversed(messages) if m.role == "user")


def _stems(text: str) -> set[str]:
    stems = set()
    for token in tokenize(text):
        for suffix, repl in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("s", "")):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[: -len(suffix)] + repl
                break
        stems.add(token)
    return stems


def _best_sentence(question: str, document: str) -> str:
    wanted = _stems(question)
    body = " ".join(line for line in document.splitlines() if line and not line.startswith("#"))
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", body) if s.strip()]
    return max(sentences, key=lambda s: (len(wanted & _stems(s)), -sentences.index(s)))


def _policy(messages: list[Message], tools: list[ToolSpec]) -> Message | str:
    last = messages[-1]
    if last.role == "tool":
        if last.is_error:
            return NOT_FOUND
        source = last.tool_call_id.removeprefix("read_") if last.tool_call_id else "?"
        return f"{_best_sentence(_question(messages), last.content)} [source: {source}]"
    context = next((m.content for m in messages if m.content.startswith("Relevant context")), "")
    match = re.search(r"\(source: ([^)]+)\)", context)
    if not match:
        return NOT_FOUND
    return FakeProvider.tool_call("read_document", f"read_{match.group(1)}", source=match.group(1))


def fake_provider() -> FakeProvider:
    return FakeProvider(_policy, model="fake-researcher")

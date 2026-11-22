# AgentFlow

[![CI](https://github.com/tatitoxt/agentflow/actions/workflows/ci.yml/badge.svg)](https://github.com/tatitoxt/agentflow/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A lightweight, typed Python (3.11+) framework for tool-calling AI agents: an explicit agent
loop, schema-validated tools, pluggable LLM providers, vector memory and JSONL tracing.
Everything runs offline by default against a deterministic scripted provider.

> Status: v0.1, a portfolio/learning project. It is not used in production and has no
> external users. See [Limitations](#limitations).

<p align="center">
  <img src="docs/demo.svg" width="880" alt="agentflow run: RAG answer + ticket triage with tool calls">
</p>

## Problem

Small tool-calling agents tend to start as glue code: a loop around a chat API, hand-written
JSON schemas for tools, string-matching for errors and `print` debugging. Large frameworks
fix that but bring big dependency trees and abstractions that are hard to read and test.
AgentFlow keeps the loop small and explicit, makes every step an inspectable event, and lets
you test agents without API keys.

## What it does today

- **Agent loop** (`agentflow.Agent`): alternates provider and tool calls until a final
  answer, a stop condition (`stop_after_tool`, `max_tool_calls`, `token_budget`, `cost_budget` or your own,
  checked after every tool call and after each provider response) or `max_steps`. Answers cut off by the provider's token limit end with
  `stop_reason="truncated"`, not `final_answer`. Long runs can be capped with
  `history_max_messages`, which drops the oldest assistant+tool groups before a
  provider request (a `history_trimmed` trace event records each trim); the head —
  instructions, retrieved context and the user's input — is always kept. With a
  `summarizer` callable the dropped messages are condensed into a persistent
  system note instead of being lost: each trim emits `history_summarized` and a
  summarizer failure falls back to plain trimming (`history_summarize_error`).
- **Tool registry** (`@tool`): JSON schema generated from type hints, `Annotated[..., Field()]`,
  Google-style docstrings or a single pydantic model parameter; arguments are validated
  (extra fields rejected) before the function runs.
- **Tool error handling**: validation errors, exceptions and unknown tools are returned to the
  model as error results so it can recover; `raise_on_tool_error=True` aborts instead.
- **Parallel tool calls**: with `max_tool_workers=N`, all calls requested in a step run
  concurrently on a thread pool; results and messages stay in call order. Stop conditions
  still apply — a mid-batch stop records the already-executed calls' real results instead
  of marking them skipped.
- **Structured output**: `agent.run_structured(input, schema=Model)` /
  `arun_structured` parses the final answer as JSON (markdown fences and
  surrounding prose tolerated), validates it against a pydantic schema, and on
  failure gives the agent a repair turn — the full conversation plus the
  validation errors — up to `max_repairs` times before raising
  `StructuredOutputError`.
- **Streaming**: `Agent.run(..., on_delta=fn)` invokes the callback with each
  assistant text fragment as it arrives (`provider_delta` trace events). Providers
  implementing `StreamingProvider` — OpenAI, Groq, Anthropic — stream over SSE;
  others get a single callback with the full response.
- **Streaming-first API**: `agent.arun_events(...)` / `agent.stream_events(...)`
  return a live iterator of the run's typed `Event`s (`run_start`,
  `provider_request`, `provider_delta`, `tool_call`, `tool_result`, `run_end`…)
  as they happen — including nested handoff runs — for progress UIs and SSE
  relays; errors re-raise after the recorded events drain.
- **Async API**: `await agent.arun(...)` mirrors `run` — the same loop driven on
  `asyncio`. Providers implementing `acomplete`/`astream` (`AsyncProvider` /
  `AsyncStreamingProvider`; OpenAI, Groq, Anthropic all do) are awaited natively
  through `httpx.AsyncClient`; sync-only providers run on a thread so the event
  loop never blocks. `run` itself bridges through a private loop in a worker
  thread, so it is legal to call from inside a running event loop, and nested
  agent runs (handoffs) keep their parent link. Cancelling the coroutine
  abandons the run and still closes the trace with `run_end`.
- **Tracing**: typed events (`run_start`, `memory_retrieval`, `provider_request`,
  `provider_delta`,
  `provider_response`, `tool_call`, `tool_result`, `tool_error`, `handoff_start`,
  `handoff_end`, `run_end`, `run_error`) streamed to sinks; JSONL export and reload.
- **Providers** behind one `Provider` protocol, all over `httpx` (no vendor SDKs):
  `OpenAIProvider` (Chat Completions), `AnthropicProvider` (Messages API), `GroqProvider` and
  `OllamaProvider` (OpenAI-compatible), with retry/backoff on 408/409/429/5xx (streaming
  included), plus `FakeProvider` (scripted, deterministic). Real providers are opt-in via env vars.
  The sync `run()` calls providers through their sync client on the caller's thread (so a
  `client=` you pass is always used and tools keep thread affinity); `arun()` awaits the async one.
- **Record/replay fixtures**: `AGENTFLOW_RECORD_FILE=path.jsonl` wraps any provider
  and appends every exchange as JSONL; `AGENTFLOW_PROVIDER=replay` (with
  `AGENTFLOW_REPLAY_FILE`) serves recorded responses offline and raises when
  the request flow diverges, so recorded runs become offline regression tests. The committed
  fixtures in `tests/fixtures/` were recorded from scripted `FakeProvider` agents, not a real model.
- **Vector memory**: `HashingEmbedder` (offline, deterministic) or `OpenAIEmbedder` (opt-in);
  `InMemoryVectorStore`, `RedisVectorStore` (Redis Stack / RediSearch KNN index, persistent),
  `PlainRedisVectorStore` (module-free Redis) or `PgVectorStore` (Postgres + pgvector).
  Retrieved snippets are injected into the agent context with their source. `PgVectorStore`
  searches exactly by default; `ensure_index()` adds an HNSW (or IVFFlat) cosine index and
  `configure_search()` tunes recall (`hnsw.ef_search` / `ivfflat.probes`) for large corpora.
  `auto_index(threshold=…)` builds it automatically once the table is big enough to benefit —
  below the threshold an exact scan is cheaper, so it stays index-free.
- **Cost accounting**: each provider response is priced via the table in
  `agentflow.pricing` (longest-prefix model matching; `register_price` for custom
  models) and accumulated into `RunResult.cost_usd`, the `provider_response` /
  `run_end` trace events and the `cost_budget` stop condition — a run halts mid-step
  once its estimated spend hits the cap. Unpriced models never accrue cost; prices are a
  hand-maintained estimate, not billing data.
- **Multi-agent handoff**: `agent.as_tool()` exposes an agent as a tool; the child run shares
  the parent's trace and records `parent_run_id`.
- **CLI**: `agentflow run examples/<name>.py` and `agentflow trace <file.jsonl>`.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/tatitoxt/agentflow.git
cd agentflow
uv sync --frozen --extra dev          # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'

# RAG over the synthetic handbook in examples/data/handbook
uv run agentflow run examples/research_assistant.py -i "How long do refunds take?"

# Ticket triage with a billing-specialist handoff, writing a JSONL trace
uv run agentflow run examples/ticket_triage.py --trace traces/triage.jsonl
uv run agentflow trace traces/triage.jsonl
```

Expected output (deterministic, no API key needed):

```text
Refunds for duplicate or incorrect charges are processed within 5 business days of approval. [source: billing.md]
Filed TCK-0001 (category=billing, priority=P2, plan=pro). Billing specialist: Duplicate charge: INV-1003 repeats INV-1002 (2026-11-01, $29.00); recommend refunding $29.00.
```

Try the failure path (unknown customer → `lookup_customer` fails → the agent recovers):

```bash
uv run agentflow run examples/ticket_triage.py \
  -i $'From: zed@unknown.test\nSubject: App is down\nEvery page returns error 500.'
```

Useful flags: `--json` (machine-readable summary), `--max-steps N`, `--provider
fake|openai|anthropic|groq|replay`. Exit codes: `0` finished, `1` error, `3` stopped early
(for example `max_steps` or `truncated`).

### Using a real model (opt-in)

```bash
export AGENTFLOW_PROVIDER=anthropic ANTHROPIC_API_KEY=...   # or openai + OPENAI_API_KEY, groq + GROQ_API_KEY
# or a local model, no key needed: AGENTFLOW_PROVIDER=ollama (OLLAMA_BASE_URL, AGENTFLOW_MODEL)
export AGENTFLOW_MODEL=...                                   # optional model override
uv run agentflow run examples/research_assistant.py -i "Is SSO available?"
```

The test suite never calls a real provider; adapters are tested against recorded wire formats
with `httpx.MockTransport`.

### Redis-backed memory

```bash
docker compose up -d --wait redis
AGENTFLOW_REDIS_URL=redis://localhost:6379/0 \
  uv run agentflow run examples/research_assistant.py -i "What is the API rate limit?"
```

### Postgres-backed memory

```bash
docker compose up -d --wait postgres
uv pip install -e '.[postgres]'
```

Then point `PgVectorStore(dim=…, url="postgresql://postgres:agentflow@localhost:55433/agentflow")`
at the container — tables and the `vector` extension are created on first use.

## Library usage

```python
from agentflow import Agent, tool
from agentflow.memory import HashingEmbedder, InMemoryVectorStore, VectorMemory
from agentflow.providers import FakeProvider


@tool
def get_weather(city: str) -> str:
    """Current weather for a city.

    Args:
        city: City name, e.g. "Lima".
    """
    return {"Lima": "18C, cloudy"}.get(city, "unknown")


memory = VectorMemory(HashingEmbedder(), InMemoryVectorStore())
memory.add("Lima is foggy in winter.", {"source": "notes.md"})

provider = FakeProvider([
    FakeProvider.tool_call("get_weather", city="Lima"),
    lambda messages, tools: f"It is {messages[-1].content} in Lima.",
])
agent = Agent("weather", provider, tools=[get_weather], memory=memory, max_steps=4)
result = agent.run("What's the weather in Lima?")
print(result.output, result.stop_reason)   # It is 18C, cloudy in Lima. final_answer
```

Swap `FakeProvider` for `OpenAIProvider.from_env()`, `AnthropicProvider.from_env()` or
`GroqProvider.from_env()` to use a real model.

## Architecture

```mermaid
flowchart LR
    CLI["agentflow CLI<br/>run / trace"] --> Example["examples/*.py<br/>build_agent() / fake_provider()"]
    Example --> Agent
    subgraph Core
        Agent["Agent loop<br/>max_steps · stop conditions"] -->|specs + messages| Provider{{"Provider protocol"}}
        Agent -->|ToolCall| Registry["ToolRegistry<br/>pydantic validation"]
        Registry -->|ToolResult / error| Agent
        Agent -->|query| Memory["VectorMemory"]
        Memory --> Embedder["HashingEmbedder · OpenAIEmbedder"]
        Memory --> Store["InMemoryVectorStore · RedisVectorStore"]
        Agent -->|events| Tracer["Tracer"] --> Sinks["MemorySink · JSONLSink"]
        Registry -.->|as_tool handoff| Child["Child Agent"]
    end
    Provider --> Fake["FakeProvider"]
    Provider --> OpenAI["OpenAI"]
    Provider --> Anthropic["Anthropic"]
    Provider --> Groq["Groq"]
    Store --> Redis[("Redis Stack<br/>RediSearch KNN")]
```

One run: `run_start` → optional `memory_retrieval` (top-k snippets injected as a system
message) → for each step `provider_request`/`provider_response`, then each requested tool is
validated and executed (`tool_call` → `tool_result` | `tool_error`) → stop on a final answer,
a stop condition or `max_steps` → `run_end`.

Code map: `src/agentflow/agent.py` (loop, stop conditions, handoff), `tools.py`,
`tracing.py`, `providers/`, `memory/`, `cli.py`. Design notes: [spec](docs/spec/agentflow.md),
[ADR 0001](docs/adr/0001-core-decisions.md).

## Tests

```bash
make lint typecheck            # ruff check, ruff format --check, mypy --strict (src + examples)
make test                      # unit + integration (Redis tests skip without a server)
make test-redis                # starts Redis Stack + Postgres/pgvector with docker compose, runs everything
make e2e                       # spawns the real `agentflow` CLI on both examples
make audit                     # pip-audit on the locked dependency set
```

| Suite | Count | What it covers |
|-------|------:|----------------|
| `tests/unit` | 126 | tools/schemas, providers' wire formats, retries and SSE streaming, sync/async loop, stop conditions, budgets, history trimming, structured output, record/replay, errors, handoff, memory, tracing, CLI, README example |
| `tests/integration` | 10 | `RedisVectorStore` KNN and agent retrieval, `PlainRedisVectorStore`, `PgVectorStore` exact + HNSW/IVFFlat search |
| `tests/e2e` | 8 | `agentflow run` / `agentflow trace` as subprocesses: outputs, exit codes, JSONL traces, tool-failure recovery, Redis-backed RAG |
| `tests/live` | 6 | opt-in real-API smoke tests (OpenAI, Anthropic); skipped without keys and not part of CI |

Counts are from `pytest --collect-only` on this commit. CI runs the 144 unit, integration and
E2E tests on Python 3.11 and 3.12 with `redis/redis-stack-server` and `pgvector/pgvector`
service containers (`AGENTFLOW_REQUIRE_REDIS=1` / `AGENTFLOW_REQUIRE_PG=1` make those tests fail
rather than skip), plus a `pip-audit --strict` job. The `tests/live` suite needs real API keys
and has not been run for this release; trigger the manual `live-smoke` workflow with
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` secrets to run it.

## Limitations

- **The demo "intelligence" is scripted.** The examples' `FakeProvider` policies are small
  rule-based functions; they show the loop, tools, retrieval and handoff working, not model
  quality. Real-provider behaviour depends on the model and was not evaluated here.
- **Real providers are not exercised against live APIs** in tests or CI; only their request /
  response and SSE formats are tested with mocked HTTP. The opt-in `tests/live` smoke tests
  exist but have not been run. `OllamaProvider` is untested against a real Ollama server.
  Vision inputs are not implemented.
- **HashingEmbedder is lexical**, not semantic: it matches overlapping words, not meaning.
- **RedisVectorStore needs Redis Stack (RediSearch)** — use `PlainRedisVectorStore` for a
  module-free Redis. The RediSearch index dimension is fixed at creation; changing embedders
  requires a new index name.
- `arun()` (async) exists for agents whose providers implement `acomplete`/`astream`;
  the synchronous `run()` is the primary API. Tools run concurrently only within a step
  (`max_tool_workers`) and only when the model requests more than one.
- `PlainRedisVectorStore` scans records in Python and `PgVectorStore` scans exactly unless
  `ensure_index()` ran — fine for small memories; use `RedisVectorStore` (RediSearch KNN)
  or an approximate index for thousands of records.
- Benchmarks are a single local micro-benchmark run (see [Benchmarks](#benchmarks)); no load,
  soak or security testing has been done.

## Benchmarks

`scripts/benchmark.py` measures framework overhead with the `FakeProvider` (no API keys) and
pgvector search latency. One run on an 8-vCPU Intel Xeon dev VM: about 2,100 two-step agent
runs/sec (0.47 ms/run) and, at 10k random 384-dim vectors, HNSW search p50 1.14 ms vs 4.83 ms
exact (about 4x). Numbers vary by machine; see [docs/benchmarks.md](docs/benchmarks.md) to
reproduce.

## Roadmap

- Run the `live-smoke` suite against real providers and publish recorded real-model fixtures.
- Vision/multimodal message parts.
- Semantic local embedder option (e.g. a small ONNX model) alongside `HashingEmbedder`.
- Packaging and a tagged release on PyPI.

## License

[MIT](LICENSE) © Fausto Pastura

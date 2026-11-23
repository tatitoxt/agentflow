# ADR 0001 — Core design decisions

Status: accepted

## Context

AgentFlow should be readable in one sitting, run fully offline in tests and CI, and avoid
vendor lock-in.

## Decisions

1. **httpx instead of vendor SDKs.** Providers are thin adapters that translate between
   AgentFlow messages and each HTTP API. This keeps the dependency tree to `httpx` and
   `pydantic`, and lets tests inject `httpx.MockTransport` to verify the exact wire format.
2. **Deterministic `FakeProvider` is the default.** Tests and examples never require API keys.
   Real providers are opt-in via `AGENTFLOW_PROVIDER` and their API-key env vars.
3. **Synchronous API.** A sync loop is easier to read and debug; async is on the roadmap.
4. **Pydantic for tool schemas.** A pydantic model is built from each tool's signature; the
   same model generates the JSON schema and validates arguments, so they cannot drift.
5. **Redis Stack (RediSearch) for persistent vectors.** It provides a KNN index with no extra
   service; the in-memory store implements the same protocol for tests and demos.
6. **Events as the single source of observability.** The loop emits typed events to sinks;
   JSONL export is one sink. No global logger configuration is required.

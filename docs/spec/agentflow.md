# AgentFlow — specification (v0.1)

## Problem

Small tool-calling agents (a support triage bot, a docs assistant) usually start as a
few hundred lines of glue: a loop around a chat-completions call, ad-hoc JSON schemas for
tools, string-matching for errors and `print` debugging. Large frameworks solve this but
bring heavy dependency trees and abstractions that are hard to read and test.

AgentFlow is a small, typed library that keeps the loop explicit and testable:
every provider call, tool call and failure is an event that can be exported as JSONL,
and every test runs offline against a deterministic scripted provider.

## Personas

- **Agent developer**: writes Python tools and wants schemas, validation and tracing for free.
- **Reviewer / maintainer**: wants to reproduce an agent run offline and read its trace.

## Requirements

| ID | Requirement |
|----|-------------|
| R1 | An agent loop that alternates provider calls and tool calls until the model returns a final answer, a stop condition fires, or `max_steps` is reached. |
| R2 | Python functions become tools with a decorator; the JSON schema comes from type hints / pydantic models; arguments are validated before the function runs. |
| R3 | Tool failures (validation errors, exceptions, unknown tools) are returned to the model as error results instead of crashing the run (configurable). |
| R4 | Structured events for each step, exportable as JSONL. |
| R5 | One provider interface with OpenAI, Anthropic and Groq adapters over httpx, selected by env vars, plus a deterministic `FakeProvider`. |
| R6 | Vector memory with a pluggable embedder (offline hashing embedder; OpenAI opt-in) and stores (in-memory; Redis Stack). Retrieved snippets are injected into the agent context. |
| R7 | An agent can be exposed as a tool so another agent can delegate (handoff). |
| R8 | A CLI `agentflow run examples/<name>.py` that runs examples offline by default. |

## Acceptance scenarios

1. **Given** an agent with a `search_docs` tool and the fake provider, **when** the user asks a
   question, **then** the run calls the tool, returns an answer citing a source and the trace
   contains `run_start`, `provider_response`, `tool_call`, `tool_result` and `run_end` events.
2. **Given** a tool call with an argument of the wrong type, **when** the loop executes it,
   **then** the function is not invoked and the model receives a validation error result.
3. **Given** a tool that raises, **when** the loop executes it, **then** a `tool_error` event is
   emitted and the run continues (or raises when `raise_on_tool_error=True`).
4. **Given** `max_steps=N` and a model that never stops calling tools, **then** the run stops
   with `stop_reason="max_steps"` after N provider calls.
5. **Given** documents stored in Redis by one process, **when** a new store instance is created
   with the same index, **then** a query returns the stored documents (persistence).
6. **Given** a triage agent with a specialist agent exposed as a tool, **when** a billing ticket
   arrives, **then** the specialist runs as a child run in the same trace and its answer is
   returned to the triage agent.
7. **Given** no API keys in the environment, **when** `agentflow run examples/<name>.py` runs,
   **then** it succeeds using the fake provider and writes a JSONL trace when `--trace` is set.

## Out of scope (v0.1)

- Streaming responses, async API, parallel tool execution.
- Long-term conversation persistence beyond vector memory.
- Prompt templating, evaluation harnesses, hosted tracing backends.
- LangChain / LlamaIndex adapters.

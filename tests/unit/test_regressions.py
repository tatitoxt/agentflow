"""Regression tests for review findings on the sync/async bridge, streaming and helpers."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from agentflow import Agent, Message, Tracer, stop_after_tool, tool
from agentflow.errors import ProviderError, StructuredOutputError, ToolExecutionError
from agentflow.providers import FakeProvider
from agentflow.providers.openai import OllamaProvider, OpenAIProvider
from agentflow.providers.replay import RecordingProvider, ReplayProvider
from agentflow.types import ToolCall


def _chat_json(content: str) -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _sse(*chunks: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def _delta(text: str, finish: str | None = None) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


def _openai(handler: Any) -> OpenAIProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIProvider(api_key="test", client=client, backoff=0)


def test_sync_run_uses_custom_client_across_repeated_runs() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=_chat_json("hi"))

    agent = Agent("a", _openai(handler))
    assert [agent.run("x").output for _ in range(3)] == ["hi", "hi", "hi"]
    assert seen == ["/v1/chat/completions"] * 3


def test_sync_run_keeps_tools_on_the_caller_thread() -> None:
    threads: list[int] = []

    @tool
    def where() -> str:
        """Record the executing thread."""
        threads.append(threading.get_ident())
        return "ok"

    provider = FakeProvider([FakeProvider.tool_call("where"), "done"])
    Agent("a", provider, tools=[where]).run("x")
    assert threads == [threading.get_ident()]


def test_stream_retries_transient_status_then_streams() -> None:
    attempts = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(attempts)
        if status != 200:
            return httpx.Response(status, text="busy")
        return httpx.Response(200, content=_sse(_delta("Hel"), _delta("lo", "stop")))

    deltas: list[str] = []
    result = Agent("a", _openai(handler)).run("x", on_delta=deltas.append)
    assert (result.output, deltas) == ("Hello", ["Hel", "lo"])


def test_stream_error_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse(_delta("par"), {"error": {"message": "overloaded"}})
        )

    with pytest.raises(ProviderError, match="stream error"):
        Agent("a", _openai(handler)).run("x", on_delta=lambda _: None)


def test_incomplete_stream_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_delta("partial")))

    with pytest.raises(ProviderError, match="finish_reason"):
        Agent("a", _openai(handler)).run("x", on_delta=lambda _: None)


def test_non_streaming_fallback_emits_provider_delta() -> None:
    tracer = Tracer()
    Agent("a", FakeProvider(["whole"]), tracer=tracer).run("x", on_delta=lambda _: None)
    assert [e.data["text"] for e in tracer.events if e.type == "provider_delta"] == ["whole"]


def test_ollama_from_env_accepts_explicit_base_url() -> None:
    provider = OllamaProvider.from_env(base_url="http://gpu-box:11434/v1")
    assert provider.base_url == "http://gpu-box:11434/v1"


def test_recording_exposes_inner_model_and_replays_across_sync_async(tmp_path: Any) -> None:
    fixture = tmp_path / "rec.jsonl"
    recorder = RecordingProvider(FakeProvider(["hi"], model="fake-1"), fixture)
    assert recorder.model == "fake-1"
    asyncio.run(Agent("a", recorder).arun("x"))
    assert Agent("a", ReplayProvider(fixture)).run("x").output == "hi"


def test_event_stream_feeds_the_agents_configured_tracer() -> None:
    tracer = Tracer()
    agent = Agent("a", FakeProvider(["ok"]), tracer=tracer)
    events = list(agent.stream_events("x"))
    assert [e.type for e in tracer.events] == [e.type for e in events]


def test_repeated_summaries_merge_into_one_note() -> None:
    @tool
    def ping(n: int) -> str:
        """Ping."""
        return f"pong {n}"

    script: list[Any] = [FakeProvider.tool_call("ping", n=i) for i in range(5)] + ["done"]
    provider = FakeProvider(script)
    agent = Agent(
        "a", provider, tools=[ping], history_max_messages=4, summarizer=lambda d: f"{len(d)}"
    )
    agent.run("x")
    last_request = provider.calls[-1][0]
    notes = [m for m in last_request if m.content.startswith("Earlier context")]
    assert len(notes) == 1
    assert len(last_request) <= 5  # limit + the merged note


class Answer(BaseModel):
    value: int


def test_structured_repair_includes_schema() -> None:
    provider = FakeProvider(["not json", '{"value": 3}'])
    result = Agent("a", provider).run_structured("x", schema=Answer)
    assert result.parsed.value == 3
    repair_prompt = provider.calls[1][0][-1].content
    assert '"value"' in repair_prompt and '"type": "integer"' in repair_prompt


def test_structured_does_not_repair_a_stopped_run() -> None:
    provider = FakeProvider([FakeProvider.tool_call("missing")])
    with pytest.raises(StructuredOutputError):
        Agent("a", provider, max_steps=1).run_structured("x", schema=Answer)
    assert len(provider.calls) == 1


def test_parallel_stop_still_raises_on_executed_tool_failure() -> None:
    @tool
    def good() -> str:
        """Succeeds."""
        return "ok"

    @tool
    def bad() -> str:
        """Fails."""
        raise RuntimeError("boom")

    message = Message.assistant(
        tool_calls=[
            ToolCall(id="1", name="good", arguments={}),
            ToolCall(id="2", name="bad", arguments={}),
        ]
    )
    agent = Agent(
        "a",
        FakeProvider([message]),
        tools=[good, bad],
        max_tool_workers=2,
        stop_conditions=[stop_after_tool("good")],
        raise_on_tool_error=True,
    )
    with pytest.raises(ToolExecutionError):
        agent.run("x")

"""``Agent.arun`` drives the same loop asynchronously; ``run`` bridges it."""

from __future__ import annotations

import asyncio

from agentflow import Agent, Message, ProviderResponse, Tracer, tool
from agentflow.providers import FakeProvider
from agentflow.types import ToolCall


@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return text


def test_arun_final_answer() -> None:
    agent = Agent("a", FakeProvider(["done"]))
    result = asyncio.run(agent.arun("hi"))
    assert result.output == "done"
    assert result.stop_reason == "final_answer"


def test_arun_tool_loop() -> None:
    provider = FakeProvider(
        [
            FakeProvider.tool_call("echo", text="ping"),
            lambda m, t: f"got {m[-1].content}",
        ]
    )
    agent = Agent("a", provider, tools=[echo])
    result = asyncio.run(agent.arun("go"))
    assert result.output == "got ping"
    assert result.steps == 2


def test_arun_parallel_tool_calls() -> None:
    calls = [ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)}) for i in range(3)]
    provider = FakeProvider(
        [
            ProviderResponse(message=Message.assistant("", calls)),
            lambda m, t: "all done",
        ]
    )
    agent = Agent("a", provider, tools=[echo], max_tool_workers=3)
    result = asyncio.run(agent.arun("go"))
    assert result.output == "all done"
    assert len(result.tool_results) == 3


def test_arun_uses_native_async_provider() -> None:
    used = {"acomplete": 0}

    class AsyncFake(FakeProvider):
        async def acomplete(self, messages, tools):
            used["acomplete"] += 1
            return self.complete(messages, tools)

    agent = Agent("a", AsyncFake(["async answer"]))
    result = asyncio.run(agent.arun("hi"))
    assert result.output == "async answer"
    assert used["acomplete"] == 1


def test_arun_streams_via_astream() -> None:
    deltas: list[str] = []

    class AsyncStreamFake(FakeProvider):
        async def astream(self, messages, tools, on_delta=None):
            response = self.complete(messages, tools)
            if on_delta:
                on_delta("chunk-")
                on_delta("end")
            return response

    agent = Agent("a", AsyncStreamFake(["streamed"]))
    result = asyncio.run(agent.arun("hi", on_delta=deltas.append))
    assert result.output == "streamed"
    assert deltas == ["chunk-", "end"]


def test_run_inside_running_loop() -> None:
    """``run`` must work even when called from a live event loop."""
    agent = Agent("a", FakeProvider(["bridged"]))

    async def invoke() -> str:
        return agent.run("hi").output

    assert asyncio.run(invoke()) == "bridged"


def test_nested_arun_keeps_parent_link() -> None:
    child = Agent("child", FakeProvider(["child out"]))
    parent = Agent(
        "parent",
        FakeProvider([FakeProvider.tool_call("delegate", task="t"), lambda m, t: "parent done"]),
        tools=[child.as_tool(name="delegate")],
    )
    tracer = Tracer()
    asyncio.run(parent.arun("go", tracer=tracer))
    child_starts = [e for e in tracer.events if e.type == "run_start" and e.agent == "child"]
    assert child_starts and child_starts[0].parent_run_id


def test_arun_events_yields_full_lifecycle() -> None:
    provider = FakeProvider(
        [
            FakeProvider.tool_call("echo", text="ping"),
            lambda m, t: "done",
        ]
    )
    agent = Agent("a", provider, tools=[echo])

    async def collect() -> list[str]:
        return [e.type async for e in agent.arun_events("go")]

    types = asyncio.run(collect())
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    assert "provider_request" in types
    assert "tool_call" in types and "tool_result" in types


def test_arun_events_final_event_carries_outcome() -> None:
    agent = Agent("a", FakeProvider(["answer"]))

    async def last() -> object:
        seen = [e async for e in agent.arun_events("hi")]
        return seen[-1]

    end = asyncio.run(last())
    assert end.data["stop_reason"] == "final_answer"
    assert "answer" in end.data["output"]


def test_arun_events_streams_provider_deltas() -> None:
    class AsyncStreamFake(FakeProvider):
        async def astream(self, messages, tools, on_delta=None):
            response = self.complete(messages, tools)
            if on_delta:
                on_delta("chunk")
            return response

    agent = Agent("a", AsyncStreamFake(["streamed"]))

    async def collect() -> list[str]:
        return [e.type async for e in agent.arun_events("hi", on_delta=lambda t: None)]

    assert "provider_delta" in asyncio.run(collect())


def test_arun_events_reraises_provider_error() -> None:
    import pytest

    from agentflow.providers.base import ProviderError

    agent = Agent("a", FakeProvider([]))  # empty script exhausts on first call

    async def drain() -> None:
        async for _ in agent.arun_events("hi"):
            pass

    with pytest.raises(ProviderError):
        asyncio.run(drain())


def test_stream_events_sync_yields_lifecycle() -> None:
    provider = FakeProvider(
        [
            FakeProvider.tool_call("echo", text="ping"),
            lambda m, t: "done",
        ]
    )
    agent = Agent("a", provider, tools=[echo])
    events = list(agent.stream_events("go"))
    types = [e.type for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert events[-1].data["stop_reason"] == "final_answer"


def test_stream_events_sync_reraises_provider_error() -> None:
    import pytest

    from agentflow.providers.base import ProviderError

    agent = Agent("a", FakeProvider([]))
    with pytest.raises(ProviderError):
        list(agent.stream_events("hi"))

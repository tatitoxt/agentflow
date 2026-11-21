from __future__ import annotations

import pytest

from agentflow import (
    Agent,
    Message,
    ProviderResponse,
    Tracer,
    Usage,
    max_tool_calls,
    stop_after_tool,
    token_budget,
    tool,
)
from agentflow.errors import ProviderError, ToolExecutionError
from agentflow.providers import FakeProvider
from agentflow.types import ToolCall, ToolSpec


@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"Lima": "18C, cloudy"}.get(city, "unknown")


@tool
def flaky(n: int) -> str:
    """Fails for negative numbers."""
    if n < 0:
        raise ValueError("negative")
    return str(n)


def last_tool_content(messages: list[Message], tools: list[ToolSpec]) -> str:
    return f"Answer: {messages[-1].content}"


def test_tool_loop_produces_final_answer_and_trace() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                message=FakeProvider.tool_call("get_weather", city="Lima"),
                usage=Usage(input_tokens=10, output_tokens=2),
            ),
            last_tool_content,
        ]
    )
    tracer = Tracer()
    agent = Agent("weather", provider, instructions="Be brief.", tools=[get_weather], tracer=tracer)
    result = agent.run("Weather in Lima?")

    assert result.output == "Answer: 18C, cloudy"
    assert result.stop_reason == "final_answer"
    assert result.steps == 2
    assert result.usage.input_tokens == 10
    assert [m.role for m in result.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert [e.type for e in tracer.events] == [
        "run_start",
        "provider_request",
        "provider_response",
        "tool_call",
        "tool_result",
        "provider_request",
        "provider_response",
        "run_end",
    ]
    assert {e.run_id for e in tracer.events} == {result.run_id}
    # the provider saw the tool schema
    assert provider.calls[0][1][0].name == "get_weather"


def test_validation_error_is_fed_back_and_model_recovers() -> None:
    provider = FakeProvider(
        [
            FakeProvider.tool_call("flaky", n="not-int"),
            FakeProvider.tool_call("flaky", n=3),
            last_tool_content,
        ]
    )
    tracer = Tracer()
    result = Agent("a", provider, tools=[flaky], tracer=tracer).run("go")
    assert result.output == "Answer: 3"
    errors = [e for e in tracer.events if e.type == "tool_error"]
    assert len(errors) == 1 and errors[0].data["error"] == "ToolValidationError"
    error_msg = provider.calls[1][0][-1]
    assert error_msg.role == "tool" and error_msg.is_error
    assert "valid integer" in error_msg.content


def test_tool_exception_continues_by_default() -> None:
    provider = FakeProvider([FakeProvider.tool_call("flaky", n=-1), last_tool_content])
    result = Agent("a", provider, tools=[flaky]).run("go")
    assert result.output.startswith("Answer: ERROR: tool 'flaky' raised ValueError: negative")
    assert result.tool_results[0].is_error


def test_tool_exception_can_abort_run() -> None:
    tracer = Tracer()
    provider = FakeProvider([FakeProvider.tool_call("flaky", n=-1)])
    agent = Agent("a", provider, tools=[flaky], raise_on_tool_error=True, tracer=tracer)
    with pytest.raises(ToolExecutionError):
        agent.run("go")
    assert [e.type for e in tracer.events][-2:] == ["run_error", "run_end"]
    assert tracer.events[-1].data["stop_reason"] == "error"


def test_unknown_tool_is_reported_to_model() -> None:
    provider = FakeProvider([FakeProvider.tool_call("nope"), last_tool_content])
    result = Agent("a", provider, tools=[flaky]).run("go")
    assert result.output == "Answer: ERROR: unknown tool: 'nope'"


def test_max_steps_stops_runaway_loop() -> None:
    provider = FakeProvider(lambda messages, tools: FakeProvider.tool_call("flaky", n=1))
    result = Agent("a", provider, tools=[flaky], max_steps=3).run("loop forever")
    assert result.stop_reason == "max_steps"
    assert result.steps == 3 and len(provider.calls) == 3
    assert result.output == ""


def test_stop_after_tool_condition() -> None:
    @tool
    def submit(answer: str) -> str:
        """Submit the final answer."""
        return answer.upper()

    provider = FakeProvider([FakeProvider.tool_call("submit", answer="done")])
    result = Agent("a", provider, tools=[submit], stop_conditions=[stop_after_tool("submit")]).run(
        "x"
    )
    assert (result.stop_reason, result.output, result.steps) == ("tool:submit", "DONE", 1)


def test_max_tool_calls_condition() -> None:
    provider = FakeProvider(lambda messages, tools: FakeProvider.tool_call("flaky", n=1))
    agent = Agent("a", provider, tools=[flaky], stop_conditions=[max_tool_calls(2)])
    result = agent.run("x")
    assert result.stop_reason == "max_tool_calls" and result.steps == 2


def test_provider_error_is_traced_and_raised() -> None:
    tracer = Tracer()
    agent = Agent("a", FakeProvider([]), tracer=tracer)
    with pytest.raises(ProviderError):
        agent.run("x")
    assert [e.type for e in tracer.events][-2:] == ["run_error", "run_end"]


def test_history_is_prepended() -> None:
    provider = FakeProvider(["ok"])
    Agent("a", provider).run("again", history=[Message.user("first"), Message.assistant("hi")])
    assert [m.content for m in provider.calls[0][0]] == ["first", "hi", "again"]


def test_invalid_max_steps() -> None:
    with pytest.raises(ValueError):
        Agent("a", FakeProvider([]), max_steps=0)


def test_handoff_runs_child_agent_in_same_trace() -> None:
    specialist = Agent(
        "billing",
        FakeProvider([lambda messages, tools: f"Refund approved for: {messages[-1].content}"]),
        instructions="You handle billing.",
    )
    handoff = specialist.as_tool(description="Send billing questions to the billing team.")
    assert handoff.name == "handoff_to_billing"
    assert handoff.spec.description == "Send billing questions to the billing team."
    assert handoff.spec.parameters["required"] == ["task"]

    triage = Agent(
        "triage",
        FakeProvider(
            [FakeProvider.tool_call("handoff_to_billing", task="double charge"), last_tool_content]
        ),
        tools=[handoff],
    )
    tracer = Tracer()
    result = triage.run("I was charged twice", tracer=tracer)
    assert result.output == "Answer: Refund approved for: double charge"

    child_runs = {e.run_id for e in tracer.events if e.agent == "billing"}
    assert len(child_runs) == 1
    child_start = next(e for e in tracer.events if e.agent == "billing" and e.type == "run_start")
    assert child_start.parent_run_id == result.run_id
    types = [(e.agent, e.type) for e in tracer.events]
    start = types.index(("triage", "handoff_start"))
    end = types.index(("triage", "handoff_end"))
    assert ("billing", "run_end") in types[start:end]


def test_handoff_reports_early_stop() -> None:
    looping = Agent(
        "worker",
        FakeProvider(lambda m, t: FakeProvider.tool_call("flaky", n=1)),
        tools=[flaky],
        max_steps=1,
    )
    output = looping.as_tool()({"task": "x"})
    assert output.startswith("[worker stopped early: max_steps]")


def _multi_call(*calls: tuple[str, dict[str, object]]) -> Message:
    return Message.assistant(
        tool_calls=[ToolCall(id=f"c{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)]
    )


def test_max_tool_calls_stops_within_a_multi_call_response() -> None:
    provider = FakeProvider([_multi_call(("flaky", {"n": 1}), ("flaky", {"n": 2}))])
    agent = Agent("a", provider, tools=[flaky], stop_conditions=[max_tool_calls(1)])
    result = agent.run("x")
    assert (result.stop_reason, result.output) == ("max_tool_calls", "1")
    assert len(result.tool_results) == 1


def test_stop_after_tool_returns_that_tools_output() -> None:
    @tool
    def submit(answer: str) -> str:
        """Submit the final answer."""
        return answer.upper()

    provider = FakeProvider([_multi_call(("submit", {"answer": "done"}), ("flaky", {"n": 7}))])
    agent = Agent("a", provider, tools=[submit, flaky], stop_conditions=[stop_after_tool("submit")])
    result = agent.run("x")
    assert (result.stop_reason, result.output) == ("tool:submit", "DONE")
    assert [r.call.name for r in result.tool_results] == ["submit"]
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c0", "c1"]
    assert tool_messages[1].content == "SKIPPED: run stopped (tool:submit)"


@pytest.mark.parametrize("provider_reason", ["length", "max_tokens"])
def test_truncated_response_is_not_a_final_answer(provider_reason: str) -> None:
    response = ProviderResponse(message=Message.assistant("partial"), stop_reason=provider_reason)
    result = Agent("a", FakeProvider([response])).run("x")
    assert (result.stop_reason, result.output) == ("truncated", "partial")


def test_handoff_child_with_own_tracer_logs_into_parent_trace() -> None:
    child_tracer, parent_tracer = Tracer(), Tracer()
    child = Agent("child", FakeProvider(["ok"]), tracer=child_tracer)
    parent = Agent(
        "parent",
        FakeProvider([FakeProvider.tool_call("handoff_to_child", task="t"), "done"]),
        tools=[child.as_tool()],
        tracer=parent_tracer,
    )
    parent.run("x")
    assert {e.agent for e in parent_tracer.events if e.type == "run_end"} == {"child", "parent"}


def test_token_budget_stops_before_pending_tool_calls() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                message=FakeProvider.tool_call("flaky", n=1),
                usage=Usage(input_tokens=100, output_tokens=5),
            ),
            last_tool_content,
        ]
    )
    agent = Agent("a", provider, tools=[flaky], stop_conditions=[token_budget(50)])
    result = agent.run("x")
    assert result.stop_reason == "token_budget"
    assert result.steps == 1 and len(provider.calls) == 1
    assert result.tool_results == []
    skipped = result.messages[-1]
    assert skipped.role == "tool" and skipped.is_error and "token_budget" in skipped.content


def test_token_budget_applies_to_final_answer_response() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                message=FakeProvider.tool_call("flaky", n=1),
                usage=Usage(input_tokens=10, output_tokens=2),
            ),
            ProviderResponse(
                message=Message.assistant("done"),
                usage=Usage(input_tokens=60, output_tokens=5),
            ),
        ]
    )
    agent = Agent("a", provider, tools=[flaky], stop_conditions=[token_budget(50)])
    result = agent.run("x")
    assert (result.stop_reason, result.output) == ("token_budget", "done")


def test_token_budget_allows_runs_under_the_limit() -> None:
    provider = FakeProvider(
        [
            ProviderResponse(
                message=FakeProvider.tool_call("flaky", n=1),
                usage=Usage(input_tokens=10, output_tokens=2),
            ),
            last_tool_content,
        ]
    )
    agent = Agent("a", provider, tools=[flaky], stop_conditions=[token_budget(1000)])
    result = agent.run("x")
    assert result.stop_reason == "final_answer" and result.output == "Answer: 1"


def test_history_max_messages_keeps_head_and_drops_whole_groups() -> None:
    provider = FakeProvider(lambda messages, tools: FakeProvider.tool_call("flaky", n=1))
    tracer = Tracer()
    agent = Agent("a", provider, tools=[flaky], max_steps=4, history_max_messages=5, tracer=tracer)
    result = agent.run("x")
    assert result.steps == 4
    seen = [len(call[0]) for call in provider.calls]
    assert all(n <= 5 for n in seen)
    sent = provider.calls[-1][0]
    assert sent[0].role == "user"  # head (system absent here) preserved
    for i, m in enumerate(sent):
        if m.role == "tool":
            assert any(c.role == "assistant" and c.tool_calls for c in sent[:i])
    assert any(e.type == "history_trimmed" for e in tracer.events)


def test_history_max_messages_never_orphans_tool_results() -> None:
    from agentflow.agent import _trim_history

    messages = [
        Message.system("s"),
        Message.user("u"),
        Message.assistant(tool_calls=[ToolCall(id="1", name="a", arguments={})]),
        Message.tool("1", "a", "r1"),
        Message.assistant(tool_calls=[ToolCall(id="2", name="b", arguments={})]),
        Message.tool("2", "b", "r2"),
        Message.assistant("final"),
    ]
    trimmed, dropped = _trim_history(messages, 5)
    assert [m.role for m in trimmed] == ["system", "user", "assistant", "tool", "assistant"]
    assert trimmed[2].tool_calls[0].id == "2"
    assert trimmed[-1].content == "final"
    assert [m.role for m in dropped] == ["assistant", "tool"]


def test_summarizer_replaces_dropped_messages_with_a_system_note() -> None:
    provider = FakeProvider(lambda messages, tools: FakeProvider.tool_call("flaky", n=1))
    tracer = Tracer()
    captured: list[list[Message]] = []

    def summarize(dropped: list[Message]) -> str:
        captured.append(list(dropped))
        return "dropped " + str(len(dropped)) + " messages about tools"

    agent = Agent(
        "a",
        provider,
        tools=[flaky],
        max_steps=4,
        history_max_messages=5,
        summarizer=summarize,
        tracer=tracer,
    )
    agent.run("x")
    assert captured, "summarizer was never called"
    assert all(m.role in {"assistant", "tool"} for batch in captured for m in batch)
    sent = provider.calls[-1][0]
    summaries = [m for m in sent if m.role == "system" and "summarized" in m.content]
    assert summaries, "no summary message reached the provider"
    assert "dropped" in summaries[-1].content
    assert any(e.type == "history_summarized" for e in tracer.events)


def test_summarizer_error_falls_back_to_plain_trimming() -> None:
    provider = FakeProvider(lambda messages, tools: FakeProvider.tool_call("flaky", n=1))
    tracer = Tracer()

    def boom(dropped: list[Message]) -> str:
        raise RuntimeError("summarizer exploded")

    agent = Agent(
        "a",
        provider,
        tools=[flaky],
        max_steps=4,
        history_max_messages=5,
        summarizer=boom,
        tracer=tracer,
    )
    result = agent.run("x")
    assert result.steps == 4
    sent = provider.calls[-1][0]
    assert all("summarized" not in m.content for m in sent)
    assert all(len(call[0]) <= 5 for call in provider.calls)
    assert any(e.type == "history_summarize_error" for e in tracer.events)


def test_parallel_tool_calls_run_concurrently_and_keep_order() -> None:
    import time

    def slow(n: int) -> str:
        time.sleep(0.15)
        return f"s{n}"

    def fast(n: int) -> str:
        return f"f{n}"

    provider = FakeProvider(
        [
            Message.assistant(
                tool_calls=[
                    ToolCall(id="a", name="slow", arguments={"n": 1}),
                    ToolCall(id="b", name="slow", arguments={"n": 2}),
                    ToolCall(id="c", name="fast", arguments={"n": 3}),
                ]
            ),
            "done",
        ]
    )
    t0 = time.monotonic()
    result = Agent("a", provider, tools=[slow, fast], max_tool_workers=3, max_steps=2).run("x")
    elapsed = time.monotonic() - t0
    assert result.stop_reason == "final_answer"
    assert [r.content for r in result.tool_results] == ["s1", "s2", "f3"]
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b", "c"]
    assert elapsed < 0.5, f"calls did not overlap (took {elapsed:.2f}s)"


def test_max_tool_workers_validation() -> None:
    provider = FakeProvider(["x"])
    try:
        Agent("a", provider, max_tool_workers=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_cost_budget_stops_run_and_reports_cost() -> None:
    from agentflow import cost_budget, register_price

    register_price("fake-1", 10.0, 20.0)
    resp = ProviderResponse(
        message=FakeProvider.tool_call("flaky", n=1),
        usage=Usage(input_tokens=10_000, output_tokens=10_000),
        model="fake-1",
    )
    # 10k in * $10/M + 10k out * $20/M = $0.30 per response.
    provider = FakeProvider([resp, resp, resp])
    result = Agent("a", provider, tools=[flaky], stop_conditions=[cost_budget(0.50)]).run("x")
    assert result.stop_reason == "cost_budget" and result.steps == 2
    assert result.cost_usd == pytest.approx(0.60)


def test_cost_is_zero_for_unpriced_models() -> None:
    provider = FakeProvider(["done"], model="no-such-model")
    result = Agent("a", provider).run("x")
    assert result.cost_usd == 0.0 and result.stop_reason == "final_answer"


def test_run_result_reports_estimated_cost() -> None:
    resp = ProviderResponse(
        message=Message.assistant("done"),
        usage=Usage(input_tokens=1_000_000, output_tokens=500_000),
        model="gpt-4o-mini",
    )
    result = Agent("a", FakeProvider([resp])).run("x")
    # $0.15/M in + $0.60/M out -> 0.15 + 0.30 = 0.45
    assert result.cost_usd == pytest.approx(0.45)


def test_on_delta_streams_response_chunks() -> None:
    provider = FakeProvider(["Hello world, this is a longer answer."])
    deltas: list[str] = []
    result = Agent("a", provider).run("x", on_delta=deltas.append)
    assert "".join(deltas) == "Hello world, this is a longer answer."
    assert result.output == "Hello world, this is a longer answer."


def test_on_delta_emits_provider_delta_trace_events() -> None:
    tracer = Tracer()
    provider = FakeProvider(["chunked output here"])
    Agent("a", provider, tracer=tracer).run("x", on_delta=lambda t: None)
    delta_events = [e for e in tracer.events if e.type == "provider_delta"]
    assert delta_events and "".join(e.data["text"] for e in delta_events) == "chunked output here"


def test_on_delta_falls_back_for_non_streaming_providers() -> None:
    class NonStreamingProvider:
        name = "non-streaming"

        def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
            return ProviderResponse(message=Message.assistant("whole answer at once"))

    deltas: list[str] = []
    result = Agent("a", NonStreamingProvider()).run("x", on_delta=deltas.append)
    assert deltas == ["whole answer at once"]
    assert result.output == "whole answer at once"

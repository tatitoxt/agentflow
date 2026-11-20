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
    assert tracer.events[-1].type == "run_error"


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
    assert [e.type for e in tracer.events][-1] == "run_error"


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

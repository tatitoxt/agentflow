"""The agent loop: provider calls, tool execution, stop conditions, memory and handoff."""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import ProviderError
from .pricing import estimate_cost
from .providers.base import Provider
from .tools import Tool, ToolRegistry, ToolResult
from .tracing import Tracer
from .types import Message, ProviderResponse, Usage

if TYPE_CHECKING:
    from .memory import VectorMemory


@dataclass
class RunState:
    """Mutable view of a run, passed to stop conditions after each step."""

    agent: str
    step: int
    messages: list[Message]
    last_response: ProviderResponse | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    step_start: int = 0

    @property
    def last_tool_results(self) -> list[ToolResult]:
        """Results executed so far for the current step's tool calls."""
        return self.tool_results[self.step_start :]


TRUNCATION_STOP_REASONS = frozenset({"length", "max_tokens"})
"""Provider stop reasons meaning the answer was cut off by the token limit."""

StopCondition = Callable[[RunState], str | None]
"""Returns a stop reason to end the run, or ``None`` to continue."""


def stop_after_tool(name: str) -> StopCondition:
    """Stop right after the tool ``name`` returns successfully; its output becomes the answer."""

    def condition(state: RunState) -> str | None:
        last = state.tool_results[-1] if state.tool_results else None
        hit = last is not None and last.call.name == name and not last.is_error
        return f"tool:{name}" if hit else None

    return condition


def max_tool_calls(limit: int) -> StopCondition:
    """Stop once ``limit`` tool calls have been executed in this run."""

    def condition(state: RunState) -> str | None:
        return "max_tool_calls" if len(state.tool_results) >= limit else None

    return condition


def token_budget(limit: int) -> StopCondition:
    """Stop once the run has consumed ``limit`` total tokens (input + output).

    Unlike tool-related conditions this is also evaluated right after each
    provider response, so a spent budget halts the run before further tool
    calls or provider requests burn more tokens.
    """

    def condition(state: RunState) -> str | None:
        total = state.usage.input_tokens + state.usage.output_tokens
        return "token_budget" if total >= limit else None

    return condition


def cost_budget(limit_usd: float) -> StopCondition:
    """Stop once the run has spent at least ``limit_usd`` USD (estimated).

    Cost is estimated per provider response from the pricing table in
    :mod:`agentflow.pricing` — models without a registered price never
    accrue cost, so an unpriced model can't burn a budget silently.
    Like :func:`token_budget` this fires right after the response, before
    further tool calls or provider requests spend more.
    """

    def condition(state: RunState) -> str | None:
        return "cost_budget" if state.cost_usd >= limit_usd else None

    return condition


@dataclass
class RunResult:
    output: str
    stop_reason: str
    steps: int
    run_id: str
    messages: list[Message]
    tool_results: list[ToolResult]
    usage: Usage
    cost_usd: float = 0.0


@dataclass(frozen=True)
class _RunContext:
    tracer: Tracer
    run_id: str
    agent: str
    state: RunState


_current_run: contextvars.ContextVar[_RunContext | None] = contextvars.ContextVar(
    "agentflow_current_run", default=None
)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _trim_history(
    messages: list[Message], limit: int
) -> tuple[list[Message], list[Message]]:
    """Keep at most ``limit`` messages: the head (system/context/user) plus the
    newest assistant+tool groups, so a kept tool result always follows the
    assistant message that requested it. Returns ``(kept, dropped)``.
    """
    if len(messages) <= limit:
        return messages, []
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.role == "assistant"), len(messages)
    )
    head, rest = messages[:first_assistant], messages[first_assistant:]
    groups: list[list[Message]] = []
    for m in rest:
        if m.role == "assistant" or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    dropped: list[Message] = []
    while groups and len(head) + sum(len(g) for g in groups) > limit:
        dropped.extend(groups.pop(0))
    return head + [m for g in groups for m in g], dropped


def _insert_summary(messages: list[Message], summary: str) -> list[Message]:
    """Insert a summary system message after the head (before the first
    assistant message), where it survives future trims.
    """
    first_assistant = next(
        (i for i, m in enumerate(messages) if m.role == "assistant"), len(messages)
    )
    note = Message.system(f"Earlier context (summarized):\n{summary}")
    return messages[:first_assistant] + [note] + messages[first_assistant:]


def _preview(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class Agent:
    """A named agent: instructions + provider + tools (+ optional vector memory)."""

    def __init__(
        self,
        name: str,
        provider: Provider,
        *,
        instructions: str = "",
        tools: Iterable[Tool | Callable[..., Any]] = (),
        memory: VectorMemory | None = None,
        memory_k: int = 3,
        max_steps: int = 8,
        history_max_messages: int | None = None,
        summarizer: Callable[[list[Message]], str] | None = None,
        max_tool_workers: int | None = None,
        stop_conditions: Sequence[StopCondition] = (),
        raise_on_tool_error: bool = False,
        tracer: Tracer | None = None,
        id_factory: Callable[[], str] = _new_run_id,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.name = name
        self.provider = provider
        self.instructions = instructions
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.memory = memory
        self.memory_k = memory_k
        self.max_steps = max_steps
        if history_max_messages is not None and history_max_messages < 4:
            raise ValueError("history_max_messages must be >= 4")
        self.history_max_messages = history_max_messages
        self.summarizer = summarizer
        if max_tool_workers is not None and max_tool_workers < 1:
            raise ValueError("max_tool_workers must be >= 1")
        self.max_tool_workers = max_tool_workers
        self.stop_conditions = list(stop_conditions)
        self.raise_on_tool_error = raise_on_tool_error
        self.tracer = tracer
        self.id_factory = id_factory

    def _context_message(self, query: str, tracer: Tracer, run_id: str) -> Message | None:
        if self.memory is None:
            return None
        hits = self.memory.search(query, k=self.memory_k)
        tracer.emit(
            "memory_retrieval",
            run_id=run_id,
            agent=self.name,
            query=query,
            hits=[{"id": h.id, "score": round(h.score, 4), "metadata": h.metadata} for h in hits],
        )
        if not hits:
            return None
        lines = ["Relevant context retrieved from memory:"]
        for i, hit in enumerate(hits, 1):
            source = hit.metadata.get("source", hit.id)
            lines.append(f"[{i}] (source: {source}) {hit.text}")
        return Message.system("\n".join(lines))

    def run(
        self,
        input: str,
        *,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Run the loop until a final answer, a stop condition or ``max_steps``.

        When ``on_delta`` is given and the provider supports streaming
        (:class:`StreamingProvider`), each assistant text fragment is passed to
        the callback as it arrives (and traced as ``provider_delta``). Providers
        without streaming fall back to one callback with the whole response.
        """
        parent = _current_run.get()
        tracer = tracer or (parent.tracer if parent else None) or self.tracer or Tracer()
        run_id = self.id_factory()
        parent_id = parent.run_id if parent else None

        def emit(type: Any, step: int = 0, **data: Any) -> None:
            tracer.emit(
                type, run_id=run_id, agent=self.name, step=step, parent_run_id=parent_id, **data
            )

        emit("run_start", input=input, max_steps=self.max_steps, tools=[t.name for t in self.tools])
        messages: list[Message] = []
        if self.instructions:
            messages.append(Message.system(self.instructions))
        messages.extend(history)
        context = self._context_message(input, tracer, run_id)
        if context is not None:
            messages.append(context)
        messages.append(Message.user(input))

        state = RunState(agent=self.name, step=0, messages=messages)
        usage = Usage()
        specs = self.tools.specs()
        output, stop_reason = "", "max_steps"
        token = _current_run.set(_RunContext(tracer, run_id, self.name, state))
        try:
            for step in range(1, self.max_steps + 1):
                state.step = step
                if self.history_max_messages is not None:
                    trimmed, dropped = _trim_history(messages, self.history_max_messages)
                    if dropped:
                        if self.summarizer is not None:
                            try:
                                summary = self.summarizer(dropped)
                            except Exception as exc:
                                emit(
                                    "history_summarize_error",
                                    step,
                                    error=type(exc).__name__,
                                    message=str(exc),
                                )
                                summary = None
                            if summary:
                                emit(
                                    "history_summarized",
                                    step,
                                    dropped=len(dropped),
                                    summary_chars=len(summary),
                                )
                                trimmed = _insert_summary(trimmed, summary)
                        emit(
                            "history_trimmed",
                            step,
                            dropped=len(dropped),
                            kept=len(trimmed),
                        )
                        messages[:] = trimmed
                        state.messages = messages
                emit("provider_request", step, provider=self.provider.name, messages=len(messages))
                try:
                    if on_delta is not None and hasattr(self.provider, "stream"):
                        def relay(text: str, *, _step: int = step) -> None:
                            emit("provider_delta", _step, text=text)
                            on_delta(text)

                        response = self.provider.stream(list(messages), specs, relay)
                    else:
                        response = self.provider.complete(list(messages), specs)
                        if on_delta is not None and response.message.content:
                            on_delta(response.message.content)
                except ProviderError as exc:
                    emit("run_error", step, error=type(exc).__name__, message=str(exc))
                    raise
                state.last_response = response
                usage.input_tokens += response.usage.input_tokens
                usage.output_tokens += response.usage.output_tokens
                state.usage = Usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                step_cost = estimate_cost(
                    response.model or getattr(self.provider, "model", None), response.usage
                )
                if step_cost is not None:
                    state.cost_usd += step_cost
                message = response.message
                messages.append(message)
                emit(
                    "provider_response",
                    step,
                    content=_preview(message.content),
                    tool_calls=[c.model_dump() for c in message.tool_calls],
                    stop_reason=response.stop_reason,
                    usage=response.usage.model_dump(),
                    cost_usd=step_cost,
                )
                reason = next((r for c in self.stop_conditions if (r := c(state))), None)
                if not message.tool_calls:
                    truncated = response.stop_reason in TRUNCATION_STOP_REASONS
                    output, stop_reason = (
                        message.content,
                        reason or ("truncated" if truncated else "final_answer"),
                    )
                    break
                if reason:
                    # A condition that fires on the response itself (e.g. a
                    # spent budget) skips every pending tool call this step.
                    output, stop_reason = message.content, reason
                    for skipped in message.tool_calls:
                        messages.append(
                            Message.tool(
                                skipped.id,
                                skipped.name,
                                f"SKIPPED: run stopped ({reason})",
                                is_error=True,
                            )
                        )
                    break
                state.step_start = len(state.tool_results)
                reason = None
                calls = message.tool_calls
                executed: list[ToolResult | None]
                if self.max_tool_workers and len(calls) > 1:
                    for c in calls:
                        emit(
                            "tool_call",
                            step,
                            id=c.id,
                            name=c.name,
                            arguments=c.arguments,
                        )
                    with ThreadPoolExecutor(max_workers=self.max_tool_workers) as pool:
                        executed = list(pool.map(self.tools.execute, calls))
                else:
                    executed = [None] * len(calls)
                for index, call in enumerate(calls):
                    result = executed[index]
                    if result is None:
                        emit(
                            "tool_call",
                            step,
                            id=call.id,
                            name=call.name,
                            arguments=call.arguments,
                        )
                        result = self.tools.execute(call)
                    state.tool_results.append(result)
                    messages.append(
                        Message.tool(call.id, call.name, result.content, is_error=result.is_error)
                    )
                    if result.is_error:
                        emit(
                            "tool_error",
                            step,
                            id=call.id,
                            name=call.name,
                            error=type(result.error).__name__,
                            message=str(result.error),
                        )
                        if self.raise_on_tool_error and result.error is not None:
                            emit("run_error", step, error=type(result.error).__name__)
                            raise result.error
                    else:
                        emit(
                            "tool_result",
                            step,
                            id=call.id,
                            name=call.name,
                            content=_preview(result.content),
                        )
                    reason = next((r for c in self.stop_conditions if (r := c(state))), None)
                    if reason:
                        output, stop_reason = result.content, reason
                        for j, skipped in enumerate(calls[index + 1 :], start=index + 1):
                            ran = executed[j]
                            if ran is not None:
                                # parallel mode: the call already ran, record its real result
                                state.tool_results.append(ran)
                                messages.append(
                                    Message.tool(
                                        skipped.id,
                                        skipped.name,
                                        ran.content,
                                        is_error=ran.is_error,
                                    )
                                )
                            else:
                                messages.append(
                                    Message.tool(
                                        skipped.id,
                                        skipped.name,
                                        f"SKIPPED: run stopped ({reason})",
                                        is_error=True,
                                    )
                                )
                        break
                if reason:
                    break
        finally:
            _current_run.reset(token)
        emit(
            "run_end",
            state.step,
            stop_reason=stop_reason,
            output=_preview(output),
            steps=state.step,
            usage=usage.model_dump(),
            cost_usd=state.cost_usd,
        )
        return RunResult(
            output=output,
            stop_reason=stop_reason,
            steps=state.step,
            run_id=run_id,
            messages=messages,
            tool_results=state.tool_results,
            usage=usage,
            cost_usd=state.cost_usd,
        )

    def as_tool(self, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a tool so another agent can delegate a task to it."""
        agent = self

        def delegate(task: str) -> str:
            parent = _current_run.get()
            if parent is not None:
                parent.tracer.emit(
                    "handoff_start",
                    run_id=parent.run_id,
                    agent=parent.agent,
                    step=parent.state.step,
                    to=agent.name,
                    task=task,
                )
            result = agent.run(task)
            if parent is not None:
                parent.tracer.emit(
                    "handoff_end",
                    run_id=parent.run_id,
                    agent=parent.agent,
                    step=parent.state.step,
                    to=agent.name,
                    child_run_id=result.run_id,
                    stop_reason=result.stop_reason,
                )
            if result.stop_reason in ("final_answer",) or result.stop_reason.startswith("tool:"):
                return result.output
            return f"[{agent.name} stopped early: {result.stop_reason}] {result.output}".strip()

        delegate.__doc__ = f"""{description or f"Delegate a task to the {agent.name} agent."}

        Args:
            task: A self-contained description of what the {agent.name} agent should do.
        """
        return Tool(delegate, name=name or f"handoff_to_{agent.name}")

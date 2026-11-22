"""The agent loop: provider calls, tool execution, stop conditions, memory and handoff."""

from __future__ import annotations

import asyncio
import contextvars
import json
import queue
import re
import sys
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

from .errors import ProviderError, StructuredOutputError
from .pricing import estimate_cost
from .providers.base import (
    AsyncProvider,
    AsyncStreamingProvider,
    Provider,
    StreamingProvider,
)
from .tools import Tool, ToolRegistry, ToolResult
from .tracing import Event, Tracer
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


_M = TypeVar("_M", bound="BaseModel")


@dataclass
class StructuredResult(Generic[_M]):
    """The outcome of :meth:`Agent.arun_structured`: the final (winning) run
    plus its schema-validated answer and how many parse attempts it took."""

    run: RunResult
    parsed: _M
    attempts: int


def _extract_json_object(text: str) -> Any:
    """Best-effort JSON extraction: whole text first, then the first {...} span."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped.strip(), flags=re.MULTILINE)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError("no JSON object found in the answer")


@dataclass(frozen=True)
class _RunContext:
    tracer: Tracer
    run_id: str
    agent: str
    state: RunState


_current_run: contextvars.ContextVar[_RunContext | None] = contextvars.ContextVar(
    "agentflow_current_run", default=None
)


_T = TypeVar("_T")


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive a coroutine to completion on a private event loop in a worker
    thread, copying contextvars so nested ``arun`` calls keep their parent.

    Using a thread (rather than ``asyncio.run`` inline) makes it legal to call
    ``Agent.run`` from inside a running event loop.
    """
    ctx = contextvars.copy_context()

    def runner() -> _T:
        return ctx.run(asyncio.run, coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(runner).result()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _trim_history(messages: list[Message], limit: int) -> tuple[list[Message], list[Message]]:
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
        """Synchronous wrapper around :meth:`arun`.

        The coroutine runs on a private event loop in a worker thread, so this
        is safe to call from inside a running event loop (e.g. a FastAPI
        handler) and from tool-execution threads for nested agent runs.
        """
        return _run_coro_sync(self.arun(input, history=history, tracer=tracer, on_delta=on_delta))

    async def arun(
        self,
        input: str,
        *,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Run the loop until a final answer, a stop condition or ``max_steps``.

        Providers implementing ``acomplete``/``astream`` (see
        :class:`AsyncProvider`) are awaited natively; sync-only providers are
        driven on a thread so they never block the event loop. Cancelling the
        coroutine abandons the run and still emits ``run_end``.

        When ``on_delta`` is given and the provider supports streaming
        (:class:`StreamingProvider`/:class:`AsyncStreamingProvider`), each
        assistant text fragment is passed to the callback as it arrives (and
        traced as ``provider_delta``). Providers without streaming fall back to
        one callback with the whole response.
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
                                summary = await asyncio.to_thread(self.summarizer, dropped)
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
                    provider = self.provider
                    if on_delta is not None and isinstance(
                        provider, AsyncStreamingProvider | StreamingProvider
                    ):

                        def relay(text: str, *, _step: int = step) -> None:
                            emit("provider_delta", _step, text=text)
                            on_delta(text)

                        if isinstance(provider, AsyncStreamingProvider):
                            response = await provider.astream(list(messages), specs, relay)
                        else:
                            response = await asyncio.to_thread(
                                provider.stream, list(messages), specs, relay
                            )
                    elif isinstance(provider, AsyncProvider):
                        response = await provider.acomplete(list(messages), specs)
                        if on_delta is not None and response.message.content:
                            on_delta(response.message.content)
                    else:
                        response = await asyncio.to_thread(provider.complete, list(messages), specs)
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
                    gate = asyncio.Semaphore(self.max_tool_workers)

                    async def run_tool(call: Any, *, _gate: asyncio.Semaphore = gate) -> ToolResult:
                        async with _gate:
                            return await asyncio.to_thread(self.tools.execute, call)

                    executed = list(await asyncio.gather(*(run_tool(c) for c in calls)))
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
                        result = await asyncio.to_thread(self.tools.execute, call)
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
            if sys.exc_info()[0] is not None:
                stop_reason = "error"
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

    def run_structured(
        self,
        input: str,
        *,
        schema: type[_M],
        max_repairs: int = 2,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> StructuredResult[_M]:
        """Synchronous wrapper around :meth:`arun_structured`."""
        return _run_coro_sync(
            self.arun_structured(
                input,
                schema=schema,
                max_repairs=max_repairs,
                history=history,
                tracer=tracer,
                on_delta=on_delta,
            )
        )

    async def arun_structured(
        self,
        input: str,
        *,
        schema: type[_M],
        max_repairs: int = 2,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> StructuredResult[_M]:
        """Run the loop, then validate the final answer against ``schema``.

        The final answer is parsed as JSON (markdown fences and surrounding
        prose are tolerated) and validated with ``schema.model_validate``. On
        failure the agent gets a repair turn — its whole conversation plus a
        user message listing the validation errors — up to ``max_repairs``
        times. A still-invalid answer raises :class:`StructuredOutputError`.
        """
        if max_repairs < 0:
            raise ValueError("max_repairs must be >= 0")
        tracer = tracer or self.tracer or Tracer()
        result = await self.arun(input, history=history, tracer=tracer, on_delta=on_delta)
        attempts = 0
        while True:
            attempts += 1
            try:
                parsed = schema.model_validate(_extract_json_object(result.output))
            except Exception as exc:
                errors = str(exc)
                tracer.emit(
                    "structured_parse_failed",
                    run_id=result.run_id,
                    agent=self.name,
                    attempt=attempts,
                    error=errors[:500],
                )
                if attempts > max_repairs:
                    raise StructuredOutputError(
                        f"final answer did not match schema {schema.__name__} "
                        f"after {attempts} attempt(s)",
                        output=result.output,
                        errors=errors,
                    ) from exc
                repair = (
                    "Your previous answer could not be validated. "
                    f"Reply with ONLY a JSON object matching schema {schema.__name__} — "
                    f"no markdown fences, no commentary.\n\nValidation errors:\n{errors[:2000]}"
                )
                tracer.emit(
                    "structured_repair",
                    run_id=result.run_id,
                    agent=self.name,
                    attempt=attempts,
                )
                result = await self.arun(repair, history=result.messages, on_delta=on_delta)
                continue
            tracer.emit(
                "structured_parsed",
                run_id=result.run_id,
                agent=self.name,
                attempt=attempts,
                schema=schema.__name__,
            )
            return StructuredResult(run=result, parsed=parsed, attempts=attempts)

    def arun_events(
        self,
        input: str,
        *,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> AsyncIterator[Event]:
        """Run the loop and yield each trace :class:`Event` as it happens.

        The stream is the same flow a :class:`Tracer` records —
        ``run_start``, ``provider_request``, ``provider_delta`` (when the
        provider streams; pass ``on_delta`` too if only the text matters),
        ``tool_call``/``tool_result``, ``run_end`` — so a consumer renders
        progress live instead of waiting on :meth:`arun`'s result. Nested
        agent runs appear inline with ``parent_run_id`` set, and the final
        ``run_end`` event carries the outcome (``stop_reason``, ``output``
        preview, ``usage``, ``cost_usd``). A provider or tool error ends the
        stream and re-raises after the recorded events have been yielded.
        """
        events: asyncio.Queue[Event | None] = asyncio.Queue()
        tracer = tracer or Tracer()
        sink = events.put_nowait
        tracer.sinks.append(sink)

        async def drive() -> None:
            try:
                await self.arun(input, history=history, tracer=tracer, on_delta=on_delta)
            finally:
                events.put_nowait(None)

        async def gen() -> AsyncIterator[Event]:
            task = asyncio.ensure_future(drive())
            try:
                while True:
                    event = await events.get()
                    if event is None:
                        break
                    yield event
                await task  # re-raise a provider/tool error once the stream drains
            finally:
                if sink in tracer.sinks:
                    tracer.sinks.remove(sink)
                if not task.done():
                    task.cancel()

        return gen()

    def stream_events(
        self,
        input: str,
        *,
        history: Sequence[Message] = (),
        tracer: Tracer | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> Iterator[Event]:
        """Synchronous counterpart of :meth:`arun_events`.

        The run executes on a worker thread while this generator yields each
        :class:`Event` live to the caller's thread, so progress can render
        without an event loop. End-of-stream and error semantics match the
        async version.
        """
        events: queue.Queue[Event | BaseException | None] = queue.Queue()
        tracer = tracer or Tracer()
        sink = events.put_nowait
        tracer.sinks.append(sink)

        def drive() -> None:
            try:
                self.run(input, history=history, tracer=tracer, on_delta=on_delta)
            except BaseException as exc:
                events.put_nowait(exc)
            finally:
                events.put_nowait(None)

        worker = threading.Thread(target=drive, daemon=True)

        def gen() -> Iterator[Event]:
            worker.start()
            try:
                while True:
                    item = events.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                if sink in tracer.sinks:
                    tracer.sinks.remove(sink)

        return gen()

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

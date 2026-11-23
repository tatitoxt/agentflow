"""Deterministic scripted provider used by tests, examples and the offline demo."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..errors import ProviderError
from ..types import Message, ProviderResponse, ToolCall, ToolSpec

Responder = Callable[[list[Message], list[ToolSpec]], "ProviderResponse | Message | str"]
ScriptItem = ProviderResponse | Message | str | Responder


class FakeProvider:
    """Replays a fixed script of responses; no network, fully deterministic.

    Each script item is one of:

    - ``str``: a final assistant answer;
    - :class:`Message` / :class:`ProviderResponse`: returned as-is;
    - a callable ``(messages, tools) -> str | Message | ProviderResponse`` that can inspect the
      conversation (for example to answer from the latest tool result).

    Passing a single callable as ``script`` uses it for every turn.
    """

    name = "fake"

    def __init__(self, script: Sequence[ScriptItem] | Responder, *, model: str = "fake-1") -> None:
        self._responder: Responder | None = None
        self._script: list[ScriptItem] = []
        if callable(script) and not isinstance(script, Sequence):
            self._responder = script
        else:
            self._script = list(script)
        self.model = model
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    @staticmethod
    def tool_call(name: str, call_id: str | None = None, /, **arguments: Any) -> Message:
        """Build an assistant message that requests a single tool call."""
        return Message.assistant(
            tool_calls=[ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments)]
        )

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        snapshot = [m.model_copy(deep=True) for m in messages]
        self.calls.append((snapshot, list(tools)))
        if self._responder is not None:
            item: ScriptItem = self._responder
        else:
            index = len(self.calls) - 1
            if index >= len(self._script):
                raise ProviderError(f"fake: script exhausted after {len(self._script)} responses")
            item = self._script[index]
        result: Any = item(snapshot, list(tools)) if callable(item) else item
        if isinstance(result, ProviderResponse):
            response = result
        elif isinstance(result, Message):
            response = ProviderResponse(message=result)
        elif isinstance(result, str):
            response = ProviderResponse(message=Message.assistant(result))
        else:
            raise ProviderError(f"fake: unsupported script item {type(result).__name__}")
        if response.stop_reason is None:
            stop = "tool_use" if response.message.tool_calls else "end_turn"
            response = response.model_copy(update={"stop_reason": stop})
        if response.model is None:
            response = response.model_copy(update={"model": self.model})
        return response

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        """Replay the scripted response, emitting the content in small chunks."""
        response = self.complete(messages, tools)
        if on_delta is not None and response.message.content:
            for i in range(0, len(response.message.content), 8):
                on_delta(response.message.content[i : i + 8])
        return response

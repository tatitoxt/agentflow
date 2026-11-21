"""Record/replay providers for regression fixtures.

``RecordingProvider`` wraps a real provider and appends every exchange to a
JSONL fixture file; ``ReplayProvider`` serves the recorded responses offline
and fails when the request flow diverges from the recording — so a change that
alters the prompt or tool flow surfaces as a test failure, not a silent diff.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import ProviderError
from ..types import Message, ProviderResponse, ToolSpec


def _request_payload(messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
    return {
        "messages": [m.model_dump(mode="json") for m in messages],
        "tools": [t.model_dump(mode="json") for t in tools],
    }


class RecordingProvider:
    """Passes through to ``inner`` and appends ``{op, request, response}`` JSONL
    records to ``path`` (created if missing)."""

    def __init__(self, inner: Any, path: str | Path) -> None:
        self._inner: Any = inner
        self._path = Path(path)
        self.name = f"recording-{inner.name}"

    def _record(
        self, op: str, messages: list[Message], tools: list[ToolSpec], resp: ProviderResponse
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "op": op,
            "request": _request_payload(messages, tools),
            "response": resp.model_dump(mode="json"),
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        resp: ProviderResponse = self._inner.complete(messages, tools)
        self._record("complete", messages, tools, resp)
        return resp

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        resp: ProviderResponse = self._inner.stream(messages, tools, on_delta)
        self._record("stream", messages, tools, resp)
        return resp

    async def acomplete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        if hasattr(self._inner, "acomplete"):
            resp = await self._inner.acomplete(messages, tools)
        else:
            resp = await asyncio.to_thread(self._inner.complete, messages, tools)
        resp_t: ProviderResponse = resp
        self._record("acomplete", messages, tools, resp_t)
        return resp_t

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        if hasattr(self._inner, "astream"):
            resp = await self._inner.astream(messages, tools, on_delta)
        elif hasattr(self._inner, "stream"):
            resp = await asyncio.to_thread(self._inner.stream, messages, tools, on_delta)
        else:
            resp = await asyncio.to_thread(self._inner.complete, messages, tools)
        resp_t: ProviderResponse = resp
        self._record("astream", messages, tools, resp_t)
        return resp_t


class ReplayProvider:
    """Replays a JSONL fixture in order. ``op`` or ``request`` divergence and
    an exhausted fixture raise :class:`ProviderError`."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        self.name = "replay"
        self._records: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
        self._pos = 0

    def _next(self, op: str, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        if self._pos >= len(self._records):
            raise ProviderError(
                f"replay exhausted: wanted {op}, fixture ended after {self._pos} calls"
            )
        rec = self._records[self._pos]
        if rec["op"] != op:
            raise ProviderError(f"replay diverged: wanted {op}, record {self._pos} is {rec['op']}")
        want = _request_payload(messages, tools)
        if rec["request"] != want:
            raise ProviderError(
                f"replay diverged: record {self._pos} request does not match the current call"
            )
        self._pos += 1
        return ProviderResponse.model_validate(rec["response"])

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        return self._next("complete", messages, tools)

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        resp = self._next("stream", messages, tools)
        if on_delta is not None and resp.message.content:
            on_delta(resp.message.content)
        return resp

    async def acomplete(self, messages: list[Message], tools: list[ToolSpec]) -> ProviderResponse:
        return self._next("acomplete", messages, tools)

    async def astream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: Callable[[str], None] | None = None,
    ) -> ProviderResponse:
        resp = self._next("astream", messages, tools)
        if on_delta is not None and resp.message.content:
            on_delta(resp.message.content)
        return resp

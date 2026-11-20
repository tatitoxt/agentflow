"""Structured run events and sinks (in-memory, JSONL)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, Field

EventType = Literal[
    "run_start",
    "memory_retrieval",
    "provider_request",
    "provider_response",
    "tool_call",
    "tool_result",
    "tool_error",
    "handoff_start",
    "handoff_end",
    "run_end",
    "run_error",
]

Sink = Callable[["Event"], None]


class Event(BaseModel):
    """One structured event. Serialized one-per-line in JSONL traces."""

    type: EventType
    run_id: str
    agent: str
    step: int = 0
    parent_run_id: str | None = None
    timestamp: float
    data: dict[str, Any] = Field(default_factory=dict)


class MemorySink:
    """Keeps events in a list; handy for tests and for exporting later."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)


class JSONLSink:
    """Appends each event as a JSON line and flushes immediately (crash-safe)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        self._fh.write(event.model_dump_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Tracer:
    """Fans events out to sinks. Always records in memory so traces can be exported."""

    def __init__(
        self, sinks: Iterable[Sink] = (), *, clock: Callable[[], float] = time.time
    ) -> None:
        self.memory = MemorySink()
        self.sinks: list[Sink] = [self.memory, *sinks]
        self.clock = clock

    @property
    def events(self) -> list[Event]:
        return self.memory.events

    def emit(
        self,
        type: EventType,
        *,
        run_id: str,
        agent: str,
        step: int = 0,
        parent_run_id: str | None = None,
        **data: Any,
    ) -> Event:
        event = Event(
            type=type,
            run_id=run_id,
            agent=agent,
            step=step,
            parent_run_id=parent_run_id,
            timestamp=self.clock(),
            data=data,
        )
        for sink in self.sinks:
            sink(event)
        return event

    def export_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for event in self.events:
                fh.write(event.model_dump_json() + "\n")
        return target


def read_jsonl(path: str | Path) -> list[Event]:
    """Load a JSONL trace back into :class:`Event` objects."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Event.model_validate(json.loads(line)) for line in lines if line.strip()]

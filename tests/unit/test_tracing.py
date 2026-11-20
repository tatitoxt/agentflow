from __future__ import annotations

import json
from pathlib import Path

from agentflow import JSONLSink, Tracer, read_jsonl


def test_tracer_records_and_fans_out(tmp_path: Path) -> None:
    seen: list[str] = []
    clock = iter([1.0, 2.0])
    tracer = Tracer([lambda e: seen.append(e.type)], clock=lambda: next(clock))
    tracer.emit("run_start", run_id="r1", agent="a", input="hi")
    tracer.emit("run_end", run_id="r1", agent="a", step=1, stop_reason="final_answer")
    assert seen == ["run_start", "run_end"]
    assert [e.timestamp for e in tracer.events] == [1.0, 2.0]
    assert tracer.events[0].data == {"input": "hi"}


def test_jsonl_sink_streams_and_export_roundtrips(tmp_path: Path) -> None:
    live = tmp_path / "live" / "trace.jsonl"
    sink = JSONLSink(live)
    tracer = Tracer([sink])
    tracer.emit("tool_call", run_id="r", agent="a", step=1, name="t", arguments={"x": 1})
    sink.close()
    line = json.loads(live.read_text().strip())
    assert line["type"] == "tool_call" and line["data"]["arguments"] == {"x": 1}

    exported = tracer.export_jsonl(tmp_path / "export.jsonl")
    events = read_jsonl(exported)
    assert events == tracer.events

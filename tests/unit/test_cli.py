from __future__ import annotations

from pathlib import Path

import pytest

from agentflow.cli import load_example, main
from agentflow.errors import AgentFlowError


def test_load_example_requires_entrypoints(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def build_agent(provider):\n    return None\n")
    with pytest.raises(AgentFlowError, match="must define a callable `fake_provider`"):
        load_example(bad)


def test_run_without_input_or_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = tmp_path / "ex.py"
    example.write_text(
        "from agentflow import Agent\n"
        "from agentflow.providers import FakeProvider\n"
        "def fake_provider():\n    return FakeProvider(['hello'])\n"
        "def build_agent(provider):\n    return Agent('ex', provider)\n"
    )
    assert main(["run", str(example)]) == 1
    assert "no --input given" in capsys.readouterr().err
    assert main(["run", str(example), "-i", "hi"]) == 0
    assert capsys.readouterr().out == "hello\n"


def test_trace_command_summarizes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = tmp_path / "ex.py"
    example.write_text(
        "from agentflow import Agent, tool\n"
        "from agentflow.providers import FakeProvider\n"
        "@tool\ndef fail(x: int) -> int:\n    raise RuntimeError('boom')\n"
        "def fake_provider():\n"
        "    return FakeProvider([FakeProvider.tool_call('fail', x=1), 'done'])\n"
        "def build_agent(provider):\n    return Agent('ex', provider, tools=[fail])\n"
    )
    trace = tmp_path / "t.jsonl"
    assert main(["run", str(example), "-i", "go", "--trace", str(trace)]) == 0
    capsys.readouterr()
    assert main(["trace", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "tool_error" in out and "fail: tool 'fail' raised RuntimeError: boom" in out
    assert out.strip().splitlines()[-1] == "8 events, 1 runs, 1 tool calls, 1 tool errors"

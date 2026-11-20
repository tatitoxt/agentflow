"""End-to-end: drive the installed `agentflow` CLI as a user would, offline (FakeProvider)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentflow import read_jsonl

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
KEY_VARS = ("AGENTFLOW_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY")


def agentflow_cmd() -> list[str]:
    script = shutil.which("agentflow", path=str(Path(sys.executable).parent))
    return [script] if script else [sys.executable, "-m", "agentflow"]


def run_cli(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k not in KEY_VARS and k != "AGENTFLOW_REDIS_URL"}
    return subprocess.run(
        [*agentflow_cmd(), *args],
        cwd=ROOT,
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_research_assistant_answers_with_citation_and_trace(tmp_path: Path) -> None:
    trace = tmp_path / "research.jsonl"
    proc = run_cli(
        "run",
        "examples/research_assistant.py",
        "-i",
        "How long do refunds take?",
        "--trace",
        str(trace),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == (
        "Refunds for duplicate or incorrect charges are processed within 5 business days of "
        "approval. [source: billing.md]"
    )
    assert "stop_reason=final_answer steps=2 tool_calls=1" in proc.stderr

    events = read_jsonl(trace)
    assert [e.type for e in events] == [
        "run_start",
        "memory_retrieval",
        "provider_request",
        "provider_response",
        "tool_call",
        "tool_result",
        "provider_request",
        "provider_response",
        "run_end",
    ]
    assert events[1].data["hits"][0]["metadata"]["source"] == "billing.md"
    assert events[4].data == {
        "id": "read_billing.md",
        "name": "read_document",
        "arguments": {"source": "billing.md"},
    }
    assert events[-1].data["stop_reason"] == "final_answer"


def test_research_assistant_declines_out_of_scope_question() -> None:
    proc = run_cli("run", "examples/research_assistant.py", "-i", "Kubernetes helm chart values?")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "I could not find this in the handbook."


@pytest.mark.redis
def test_research_assistant_with_redis_memory(redis_url: str) -> None:
    proc = run_cli(
        "run",
        "examples/research_assistant.py",
        "-i",
        "What is the API rate limit?",
        "--json",
        AGENTFLOW_REDIS_URL=redis_url,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["output"].endswith("[source: api.md]")
    assert "100 requests per minute" in summary["output"]


def test_ticket_triage_hands_off_to_billing_specialist(tmp_path: Path) -> None:
    trace = tmp_path / "triage.jsonl"
    proc = run_cli("run", "examples/ticket_triage.py", "--trace", str(trace))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == (
        "Filed TCK-0001 (category=billing, priority=P2, plan=pro). Billing specialist: "
        "Duplicate charge: INV-1003 repeats INV-1002 (2026-11-01, $29.00); "
        "recommend refunding $29.00."
    )

    events = read_jsonl(trace)
    runs = {e.agent: e.run_id for e in events if e.type == "run_start"}
    assert set(runs) == {"triage", "billing"}
    child_start = next(e for e in events if e.type == "run_start" and e.agent == "billing")
    assert child_start.parent_run_id == runs["triage"]
    tool_calls = [(e.agent, e.data["name"]) for e in events if e.type == "tool_call"]
    assert tool_calls == [
        ("triage", "lookup_customer"),
        ("triage", "handoff_to_billing"),
        ("billing", "get_invoices"),
        ("triage", "create_ticket"),
    ]
    ticket = next(e for e in events if e.type == "tool_call" and e.data["name"] == "create_ticket")
    assert ticket.data["arguments"]["priority"] == "P2"

    summary = run_cli("trace", str(trace))
    assert summary.returncode == 0
    assert summary.stdout.strip().splitlines()[-1] == (
        "26 events, 2 runs, 4 tool calls, 0 tool errors"
    )


def test_ticket_triage_recovers_from_tool_failure(tmp_path: Path) -> None:
    trace = tmp_path / "unknown.jsonl"
    ticket = "From: zed@unknown.test\nSubject: App is down\nEvery page returns error 500."
    proc = run_cli("run", "examples/ticket_triage.py", "-i", ticket, "--trace", str(trace))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == (
        "Filed TCK-0001 (category=bug, priority=P1, plan=unknown). "
        "Sender is not a known customer; marked as unverified."
    )
    events = read_jsonl(trace)
    errors = [e for e in events if e.type == "tool_error"]
    assert len(errors) == 1
    assert errors[0].data["name"] == "lookup_customer"
    assert errors[0].data["error"] == "ToolExecutionError"
    assert "no customer with email zed@unknown.test" in errors[0].data["message"]
    assert events[-1].type == "run_end" and events[-1].data["stop_reason"] == "final_answer"


def test_max_steps_exits_with_code_3() -> None:
    proc = run_cli("run", "examples/ticket_triage.py", "--max-steps", "1", "--json")
    assert proc.returncode == 3
    summary = json.loads(proc.stdout)
    assert summary["stop_reason"] == "max_steps"
    assert summary["steps"] == 1 and summary["tool_calls"] == ["lookup_customer"]


def test_real_provider_requires_api_key() -> None:
    proc = run_cli("run", "examples/ticket_triage.py", "--provider", "anthropic")
    assert proc.returncode == 1
    assert "set ANTHROPIC_API_KEY" in proc.stderr
    assert proc.stdout == ""


def test_missing_example_is_reported() -> None:
    proc = run_cli("run", "examples/does_not_exist.py")
    assert proc.returncode == 1
    assert "example not found" in proc.stderr

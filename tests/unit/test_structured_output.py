"""run_structured / arun_structured: schema validation + repair turns."""

import pytest
from pydantic import BaseModel

from agentflow import Agent, StructuredOutputError
from agentflow.providers.fake import FakeProvider


class Ticket(BaseModel):
    title: str
    severity: int
    resolved: bool


def test_structured_output_parses_clean_json():
    agent = Agent(
        "triage",
        FakeProvider(['{"title": "login broke", "severity": 2, "resolved": false}']),
    )
    out = agent.run_structured("triage this", schema=Ticket)
    assert out.parsed == Ticket(title="login broke", severity=2, resolved=False)
    assert out.attempts == 1


def test_structured_output_tolerates_fences_and_prose():
    agent = Agent(
        "triage",
        FakeProvider(
            ['Here you go:\n```json\n{"title": "x", "severity": 1, "resolved": true}\n```\nDone.']
        ),
    )
    out = agent.run_structured("triage", schema=Ticket)
    assert out.parsed.resolved is True
    assert out.attempts == 1


def test_structured_output_repairs_bad_json():
    agent = Agent(
        "triage",
        FakeProvider(
            [
                "not json at all",
                '{"title": "fixed", "severity": 3, "resolved": false}',
            ]
        ),
    )
    out = agent.run_structured("triage", schema=Ticket)
    assert out.attempts == 2
    assert out.parsed.title == "fixed"
    # The repair turn continued the same conversation: the repair prompt is
    # the last user message on run two.
    provider = agent.provider
    second_run_messages = provider.calls[1][0]
    assert any("could not be validated" in m.content for m in second_run_messages)


def test_structured_output_repairs_schema_mismatch():
    agent = Agent(
        "triage",
        FakeProvider(
            [
                '{"title": "x", "severity": "high"}',  # wrong type
                '{"title": "x", "severity": 4, "resolved": false}',
            ]
        ),
    )
    out = agent.run_structured("triage", schema=Ticket)
    assert out.parsed.severity == 4


def test_structured_output_gives_up_after_max_repairs():
    agent = Agent("triage", FakeProvider(["nope", "still nope", "and again"]))
    with pytest.raises(StructuredOutputError) as exc:
        agent.run_structured("triage", schema=Ticket, max_repairs=2)
    assert exc.value.output == "and again"


def test_structured_output_zero_repairs_fails_fast():
    agent = Agent("triage", FakeProvider(["nope", "unused"]))
    with pytest.raises(StructuredOutputError):
        agent.run_structured("triage", schema=Ticket, max_repairs=0)


def test_arun_structured_async():
    import asyncio

    agent = Agent(
        "triage",
        FakeProvider(['{"title": "async", "severity": 1, "resolved": false}']),
    )
    out = asyncio.run(agent.arun_structured("triage", schema=Ticket))
    assert out.parsed.title == "async"
    assert out.run.stop_reason == "final_answer"

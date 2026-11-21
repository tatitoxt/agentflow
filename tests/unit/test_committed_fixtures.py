"""Replay committed record/replay fixtures and assert recorded expectations.

Fixtures under tests/fixtures/ were produced by scripts/record_fixtures.py;
ReplayProvider raises if the request flow diverges from the recording, so a
change to the agent's prompt or tool-call flow fails here, not silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentflow import Agent
from agentflow.providers.replay import ReplayProvider

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def get_weather(city: str) -> str:
    """Return the weather for a city.

    Args:
        city: City name.
    """
    return f"{city}: 18C, cloudy"


def add(a: int, b: int) -> str:
    """Add two numbers.

    Args:
        a: First addend.
        b: Second addend.
    """
    return str(a + b)


def mul(a: int, b: int) -> str:
    """Multiply two numbers.

    Args:
        a: First factor.
        b: Second factor.
    """
    return str(a * b)


TOOLS = {"get_weather": get_weather, "add": add, "mul": mul}


@pytest.mark.parametrize(
    "fixture", sorted(p.stem for p in FIXTURES.glob("*.jsonl"))
)
def test_committed_fixture_replays(fixture: str) -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest[fixture]
    agent = Agent(
        "fixture",
        ReplayProvider(FIXTURES / f"{fixture}.jsonl"),
        tools=[TOOLS[name] for name in meta["tools"]],
        **meta["agent_kwargs"],
    )
    result = agent.run(meta["prompt"])
    assert result.output == meta["output"]
    assert result.stop_reason == meta["stop_reason"]
    assert result.steps == meta["steps"]

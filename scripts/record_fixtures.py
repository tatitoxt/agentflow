"""Regenerate the committed replay fixtures under tests/fixtures/.

Runs scripted FakeProvider agents through RecordingProvider so the JSONL
fixtures match exactly what the agent loop sends. The unit test in
tests/unit/test_committed_fixtures.py replays them via ReplayProvider and
asserts the manifest expectations — any change to the prompt/flow the agent
emits surfaces as a replay divergence failure.

Usage: .venv/bin/python scripts/record_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentflow import Agent  # noqa: E402
from agentflow.providers.fake import FakeProvider  # noqa: E402
from agentflow.providers.replay import RecordingProvider  # noqa: E402
from agentflow.types import ProviderResponse  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


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


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}

    def record(name: str, fake: FakeProvider, agent_kwargs: dict, prompt: str) -> None:
        path = FIXTURES / f"{name}.jsonl"
        path.unlink(missing_ok=True)
        agent = Agent("fixture", RecordingProvider(fake, path), **agent_kwargs)
        result = agent.run(prompt)
        manifest[name] = {
            "output": result.output,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "prompt": prompt,
            "agent_kwargs": {
                k: v for k, v in agent_kwargs.items() if isinstance(v, str | int | bool)
            },
            "tools": [t.__name__ for t in agent_kwargs.get("tools", [])],
        }

    record(
        "simple_chat",
        FakeProvider(["Hello! How can I help?"]),
        {"instructions": "You are a fixture agent."},
        "Say hi",
    )

    record(
        "tool_loop",
        FakeProvider(
            [
                ProviderResponse(message=FakeProvider.tool_call("get_weather", city="Lima")),
                lambda messages, tools: f"Answer: {messages[-1].content}",
            ]
        ),
        {"instructions": "Be brief.", "tools": [get_weather]},
        "Weather in Lima?",
    )

    record(
        "parallel_tools",
        FakeProvider(
            [
                ProviderResponse(
                    message=FakeProvider.tool_call(
                        "batch",
                        call_id="batch-1",
                    ).model_copy(
                        update={
                            "tool_calls": [
                                FakeProvider.tool_call("add", call_id="c-add", a=2, b=3)
                                .tool_calls[0],
                                FakeProvider.tool_call("mul", call_id="c-mul", a=4, b=5)
                                .tool_calls[0],
                            ]
                        }
                    )
                ),
                lambda messages, tools: "Sum=5, product=20",
            ]
        ),
        {"tools": [add, mul], "max_tool_workers": 2},
        "Compute 2+3 and 4*5.",
    )

    (FIXTURES / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, meta in manifest.items():
        print(f"{name}: {meta['steps']} steps -> {meta['output']}")


if __name__ == "__main__":
    main()

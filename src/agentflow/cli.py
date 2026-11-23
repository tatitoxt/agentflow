"""Command-line entrypoint: ``agentflow run examples/<name>.py``."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from . import __version__
from .agent import Agent
from .errors import AgentFlowError
from .providers import PROVIDER_NAMES, provider_from_env
from .tracing import JSONLSink, Tracer, read_jsonl

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STOPPED_EARLY = 3


def load_example(path: str | Path) -> ModuleType:
    """Import an example file. It must define ``build_agent(provider)`` and ``fake_provider()``."""
    file = Path(path)
    if not file.is_file():
        raise AgentFlowError(f"example not found: {file}")
    spec = importlib.util.spec_from_file_location(f"agentflow_example_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise AgentFlowError(f"cannot import {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    for attr in ("build_agent", "fake_provider"):
        if not callable(getattr(module, attr, None)):
            raise AgentFlowError(f"{file} must define a callable `{attr}`")
    return module


def _run(args: argparse.Namespace) -> int:
    module = load_example(args.example)
    text = args.input or getattr(module, "DEFAULT_INPUT", None)
    if not text:
        raise AgentFlowError("no --input given and the example has no DEFAULT_INPUT")
    provider = provider_from_env(args.provider, fake=module.fake_provider())
    agent: Agent = module.build_agent(provider)
    if args.max_steps is not None:
        agent.max_steps = args.max_steps
    sink = JSONLSink(args.trace) if args.trace else None
    if sink is not None:
        sink.path.write_text("", encoding="utf-8")
    tracer = Tracer([sink] if sink else [])
    try:
        result = agent.run(text, tracer=tracer)
    finally:
        if sink is not None:
            sink.close()
    finished = result.stop_reason == "final_answer" or result.stop_reason.startswith("tool:")
    if args.json:
        payload = {
            "output": result.output,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "run_id": result.run_id,
            "provider": provider.name,
            "tool_calls": [r.call.name for r in result.tool_results],
            "tool_errors": sum(r.is_error for r in result.tool_results),
            "usage": result.usage.model_dump(),
            "events": len(tracer.events),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.output)
    print(
        f"[agentflow] agent={agent.name} provider={provider.name} stop_reason={result.stop_reason}"
        f" steps={result.steps} tool_calls={len(result.tool_results)} events={len(tracer.events)}"
        + (f" trace={args.trace}" if args.trace else ""),
        file=sys.stderr,
    )
    return EXIT_OK if finished else EXIT_STOPPED_EARLY


def _trace(args: argparse.Namespace) -> int:
    events = read_jsonl(args.path)
    for e in events:
        detail = ""
        if e.type == "tool_call":
            detail = f"{e.data['name']}({json.dumps(e.data['arguments'])})"
        elif e.type in ("tool_result", "tool_error"):
            detail = f"{e.data['name']}: {e.data.get('content') or e.data.get('message')}"
        elif e.type == "run_end":
            detail = f"stop_reason={e.data['stop_reason']}"
        elif e.type in ("handoff_start", "handoff_end"):
            detail = f"-> {e.data['to']}"
        print(f"{e.run_id} {e.agent:<10} step={e.step} {e.type:<17} {detail[:120]}".rstrip())
    counts = Counter(e.type for e in events)
    print(
        f"{len(events)} events, {len({e.run_id for e in events})} runs, "
        f"{counts['tool_call']} tool calls, {counts['tool_error']} tool errors"
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentflow", description=__doc__)
    parser.add_argument("--version", action="version", version=f"agentflow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an example agent file")
    run.add_argument("example", help="path to an example, e.g. examples/ticket_triage.py")
    run.add_argument("-i", "--input", help="user input (defaults to the example's DEFAULT_INPUT)")
    run.add_argument(
        "-p",
        "--provider",
        choices=PROVIDER_NAMES,
        help="provider to use (default: $AGENTFLOW_PROVIDER or 'fake')",
    )
    run.add_argument("--trace", help="write a JSONL trace to this path")
    run.add_argument("--max-steps", type=int, help="override the agent's max_steps")
    run.add_argument("--json", action="store_true", help="print a JSON summary instead of text")
    run.set_defaults(func=_run)

    trace = sub.add_parser("trace", help="summarize a JSONL trace file")
    trace.add_argument("path")
    trace.set_defaults(func=_trace)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except AgentFlowError as exc:
        print(f"agentflow: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

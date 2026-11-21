"""Reproducible AgentFlow benchmarks — framework overhead and vector memory.

Usage:
    uv run python scripts/benchmark.py [--runs 200] [--docs 10000]
                                       [--pg-url postgresql://...] [--json]

Measures, no API keys needed:
  - agent loop: runs/sec and steps/sec through the full loop (message
    assembly, tool dispatch, tracing, stop conditions) with FakeProvider
  - PgVectorStore: upsert throughput and exact vs HNSW search latency
    (p50/p99) over N documents — requires --pg-url
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

from agentflow import Agent, tool
from agentflow.providers.fake import FakeProvider


@tool
def lookup(term: str) -> str:
    """Return a canned fact."""
    return f"fact:{term}"


def bench_agent_loop(runs: int) -> dict:
    """Each run is a 2-step agent: one tool call, then a final answer."""

    def responder(messages, _tools):
        if any(m.role == "tool" for m in messages):
            return "done"
        return FakeProvider.tool_call("lookup", term="x")

    provider = FakeProvider(responder)
    agent = Agent("bench", provider=provider, instructions="i", tools=[lookup], max_steps=4)

    start = time.perf_counter()
    for _ in range(runs):
        result = agent.run("q")
        assert result.stop_reason == "final_answer", result.stop_reason
    elapsed = time.perf_counter() - start

    return {
        "runs": runs,
        "runs_per_sec": round(runs / elapsed, 1),
        "steps_per_sec": round(runs * 2 / elapsed, 1),
        "ms_per_run": round(elapsed / runs * 1000, 2),
    }


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def bench_pgvector(url: str, docs: int, dim: int = 384) -> dict:
    from agentflow.memory.stores import PgVectorStore, Record

    rng = random.Random(42)
    store = PgVectorStore(dim=dim, url=url, index_name=f"bench_{int(time.time())}")
    try:
        records = [
            Record(
                id=f"doc-{i}",
                text=f"document {i}",
                vector=[rng.gauss(0, 1) for _ in range(dim)],
            )
            for i in range(docs)
        ]

        start = time.perf_counter()
        for off in range(0, docs, 500):
            store.upsert(records[off : off + 500])
        upsert_ms = time.perf_counter() - start

        def timed_searches() -> list[float]:
            lat = []
            for _ in range(50):
                q = [rng.gauss(0, 1) for _ in range(dim)]
                t0 = time.perf_counter()
                store.search(q, k=10)
                lat.append((time.perf_counter() - t0) * 1000)
            return lat

        exact = timed_searches()
        store.ensure_index("hnsw")
        store.configure_search(ef_search=40)
        hnsw = timed_searches()

        return {
            "docs": docs,
            "dim": dim,
            "upsert_per_sec": round(docs / upsert_ms, 0),
            "exact_search_ms": {
                "p50": round(percentile(exact, 0.5), 2),
                "p99": round(percentile(exact, 0.99), 2),
            },
            "hnsw_search_ms": {
                "p50": round(percentile(hnsw, 0.5), 2),
                "p99": round(percentile(hnsw, 0.99), 2),
            },
            "hnsw_speedup_p50": round(percentile(exact, 0.5) / percentile(hnsw, 0.5), 1),
        }
    finally:
        store.clear()
        store.conn.execute(f'DROP TABLE IF EXISTS "{store.table}"')
        store.conn.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--docs", type=int, default=10000)
    p.add_argument("--pg-url", default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out: dict = {"agent_loop": bench_agent_loop(args.runs)}
    if args.pg_url:
        out["pgvector"] = bench_pgvector(args.pg_url, args.docs)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        al = out["agent_loop"]
        print(f"agent loop: {al['runs_per_sec']} runs/sec "
              f"({al['ms_per_run']} ms/run, {al['steps_per_sec']} steps/sec)")
        if "pgvector" in out:
            pg = out["pgvector"]
            print(f"pgvector {pg['docs']} docs: {pg['upsert_per_sec']:.0f} upserts/sec, "
                  f"search p50 exact {pg['exact_search_ms']['p50']} ms "
                  f"vs hnsw {pg['hnsw_search_ms']['p50']} ms "
                  f"({pg['hnsw_speedup_p50']}x faster)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

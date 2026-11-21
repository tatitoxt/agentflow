# Benchmarks

Measured with `scripts/benchmark.py` — no API keys needed for the agent-loop
bench; the memory bench runs against a real Postgres + pgvector.

**How to reproduce**

```sh
python scripts/benchmark.py --runs 300 \
    --docs 10000 --pg-url postgresql://postgres:agentflow@host:5433/agentflow
```

**Results (2026-11-21, local Postgres 17 + pgvector)**

| Metric | Value |
|---|---:|
| Agent loop (2-step tool-call run, FakeProvider) | **3,787 runs/sec** — 0.26 ms/run |
| Loop steps/sec | ~7,575 |
| pgvector upserts (dim 384, batches of 500) | 2,708 records/sec |
| Exact k-NN search over 10k docs (k=10) | 3.85 ms p50 / 5.4 ms p99 |
| HNSW search, same corpus | **1.18 ms p50 — 3.3× faster** |

**Reading the numbers**

- The loop number isolates framework overhead: message assembly, tool
  dispatch, tracing, stop-condition checks and cost accounting. A live
  provider adds its own latency on top.
- HNSW search speedup grows with corpus size — at 10k vectors the gain is
  already 3.3×; tune recall with `configure_search(ef_search=…)`.
- Numbers are machine-dependent; they are a reproducible baseline, not a
  marketing figure.

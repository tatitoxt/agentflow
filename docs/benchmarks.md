# Benchmarks

Measured with `scripts/benchmark.py`. The agent-loop bench needs no API keys (it uses the
`FakeProvider`); the memory bench runs against a real Postgres + pgvector.

**How to reproduce**

```sh
docker compose up -d --wait postgres
uv run --frozen python scripts/benchmark.py --runs 300 \
    --docs 10000 --pg-url postgresql://postgres:agentflow@localhost:55433/agentflow
```

**Results (single run, 2026-11-22, 8-vCPU Intel Xeon dev VM, Python 3.11, pgvector/pgvector:pg17 in Docker)**

| Metric | Value |
|---|---:|
| Agent loop (2-step tool-call run, FakeProvider, sync `run()`) | 2,124 runs/sec (0.47 ms/run) |
| Loop steps/sec | 4,248 |
| pgvector upserts (dim 384) | 2,709 records/sec |
| Exact k-NN search over 10k docs (k=10), p50 | 4.83 ms |
| HNSW search, same corpus, p50 | 1.14 ms (about 4.2x faster) |

**Reading the numbers**

- The loop number isolates framework overhead: message assembly, tool dispatch, tracing,
  stop-condition checks and cost accounting. A live provider adds its own network latency.
- Vectors are random Gaussian, not real embeddings; HNSW recall was not measured.
- Numbers are machine-dependent and come from one run; treat them as a reproducible
  baseline, not a performance claim.

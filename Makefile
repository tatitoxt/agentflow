.PHONY: install lint typecheck test test-redis e2e redis-up redis-down audit demo check

install:
	uv sync --frozen --extra dev

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

typecheck:
	uv run --frozen mypy

test:
	uv run --frozen pytest tests/unit tests/integration

redis-up:
	docker compose up -d --wait redis postgres

redis-down:
	docker compose down

test-redis: redis-up
	AGENTFLOW_TEST_REDIS_URL=redis://localhost:6379/0 AGENTFLOW_REQUIRE_REDIS=1 \
	AGENTFLOW_TEST_PG_URL=postgresql://postgres:agentflow@localhost:55433/agentflow AGENTFLOW_REQUIRE_PG=1 \
	uv run --frozen pytest tests/unit tests/integration tests/e2e

e2e:
	uv run --frozen pytest tests/e2e

audit:
	uv export --frozen --extra dev --no-hashes --no-emit-project > requirements-audit.txt
	uv run --frozen pip-audit -r requirements-audit.txt --no-deps --disable-pip --strict

demo:
	uv run --frozen agentflow run examples/research_assistant.py -i "How long do refunds take?"
	uv run --frozen agentflow run examples/ticket_triage.py --trace traces/triage.jsonl
	uv run --frozen agentflow trace traces/triage.jsonl

check: lint typecheck test e2e

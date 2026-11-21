"""Model pricing: maps provider model names to per-token USD costs.

Used by :func:`~agentflow.cost_budget` to stop a run once it has spent a
configurable amount of money, and exposed on ``RunResult.cost_usd`` so runs
are billable/accountable.

Prices are in USD per **1M tokens** (input, output), keyed by the model name
the provider reports. ``register_price`` adds or overrides entries for
custom/self-hosted models; unknown models simply don't accrue cost
(``estimate_cost`` returns ``None``), so budgets are never charged against
made-up prices.
"""

from __future__ import annotations

from .types import Usage

# USD per 1M tokens: {model_prefix: (input, output)}.
# Source: public list prices (OpenAI / Anthropic / Groq) at time of writing.
# These are estimates for budgeting, not billing truth.
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    # Groq-hosted (per-token billing)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "meta-llama/llama-4-scout": (0.11, 0.34),
    "meta-llama/llama-4-maverick": (0.20, 0.60),
}


def register_price(model: str, input_per_million: float, output_per_million: float) -> None:
    """Register or override the price for a model name (or prefix)."""
    PRICES[model] = (input_per_million, output_per_million)


def _match(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    # Longest prefix wins so "gpt-4.1-mini-2025-04-14" hits "gpt-4.1-mini"
    # before "gpt-4.1".
    best: tuple[float, float] | None = None
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]
    return best


def estimate_cost(model: str | None, usage: Usage) -> float | None:
    """USD cost of one response's usage, or ``None`` when the model is unpriced."""
    if not model:
        return None
    prices = _match(model)
    if prices is None:
        return None
    pin, pout = prices
    return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000

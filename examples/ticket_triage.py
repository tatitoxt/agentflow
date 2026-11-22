"""Support ticket triage with a billing specialist handoff.

The triage agent looks up the customer, delegates billing questions to a ``billing``
specialist agent (exposed as a tool via ``Agent.as_tool``) and files a ticket with a
category and priority. Unknown customers make ``lookup_customer`` fail; the agent recovers
and files an unverified ticket. All data is synthetic (``examples/data/customers.json``).

    agentflow run examples/ticket_triage.py --trace traces/triage.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from agentflow import Agent, Message, ToolSpec, tool
from agentflow.providers import FakeProvider, Provider

CUSTOMERS: dict[str, dict[str, Any]] = json.loads(
    (Path(__file__).parent / "data" / "customers.json").read_text(encoding="utf-8")
)
DEFAULT_INPUT = (
    "From: ana@example.com\n"
    "Subject: Charged twice this month\n"
    "Hi, I see two charges for my Pro subscription on November 1st. Can you refund one?"
)

TRIAGE_INSTRUCTIONS = """You triage support emails for Acme Notes.
1. Call lookup_customer with the sender email.
2. For billing questions, delegate to the billing specialist with handoff_to_billing.
3. Call create_ticket with a category, a priority (P1 enterprise or outage, P2 paid plans,
   P3 otherwise) and a one-line summary.
4. Reply with the ticket id, category, priority and any specialist notes."""

BILLING_INSTRUCTIONS = """You are the billing specialist. Inspect the customer's invoices with
get_invoices and report duplicate charges and the refund amount, in one sentence."""

Category = Literal["billing", "bug", "account", "other"]
Priority = Literal["P1", "P2", "P3"]

TICKETS: list[dict[str, str]] = []


@tool
def lookup_customer(email: str) -> dict[str, str]:
    """Look up a customer's name and plan by email address.

    Args:
        email: The sender's email address.
    """
    customer = CUSTOMERS.get(email.lower())
    if customer is None:
        raise LookupError(f"no customer with email {email}")
    return {"email": email.lower(), "name": customer["name"], "plan": customer["plan"]}


@tool
def get_invoices(email: str) -> list[dict[str, Any]]:
    """List a customer's invoices (id, date, amount in USD).

    Args:
        email: The customer's email address.
    """
    customer = CUSTOMERS.get(email.lower())
    if customer is None:
        raise LookupError(f"no customer with email {email}")
    invoices: list[dict[str, Any]] = customer["invoices"]
    return invoices


@tool
def create_ticket(category: Category, priority: Priority, summary: str, email: str) -> str:
    """File a support ticket and return its id.

    Args:
        category: One of billing, bug, account, other.
        priority: P1 (urgent), P2 (normal) or P3 (low).
        summary: One-line summary of the request.
        email: Customer email address.
    """
    ticket_id = f"TCK-{len(TICKETS) + 1:04d}"
    TICKETS.append(
        {
            "id": ticket_id,
            "category": category,
            "priority": priority,
            "summary": summary,
            "email": email,
        }
    )
    return ticket_id


def build_billing_agent(provider: Provider) -> Agent:
    return Agent(
        "billing",
        provider,
        instructions=BILLING_INSTRUCTIONS,
        tools=[get_invoices],
        max_steps=3,
    )


def build_agent(provider: Provider, billing_provider: Provider | None = None) -> Agent:
    TICKETS.clear()
    if billing_provider is None:
        billing_provider = (
            FakeProvider(billing_policy, model="fake-billing")
            if isinstance(getattr(provider, "inner", provider), FakeProvider)
            else provider
        )
    billing = build_billing_agent(billing_provider)
    return Agent(
        "triage",
        provider,
        instructions=TRIAGE_INSTRUCTIONS,
        tools=[
            lookup_customer,
            billing.as_tool(description="Delegate a billing question to the billing specialist."),
            create_ticket,
        ],
        max_steps=6,
    )


# --- Deterministic policies used by the FakeProvider ----------------------------------------

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("charge", "charged", "refund", "invoice", "billing", "payment"),
    "bug": ("error", "crash", "bug", "broken", "down", "outage", "500"),
    "account": ("password", "login", "log in", "2fa", "sso", "invite"),
}


def _email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    return match.group(0).lower() if match else "unknown@invalid"


def _category(text: str) -> str:
    lowered = text.lower()
    for category, words in _KEYWORDS.items():
        if any(w in lowered for w in words):
            return category
    return "other"


def _subject(text: str) -> str:
    match = re.search(r"^Subject:\s*(.+)$", text, re.M)
    return match.group(1).strip() if match else text.strip().splitlines()[0][:80]


def _tool_output(messages: list[Message], name: str) -> Message | None:
    return next((m for m in reversed(messages) if m.role == "tool" and m.name == name), None)


def triage_policy(messages: list[Message], tools: list[ToolSpec]) -> Message | str:
    ticket = next(m.content for m in messages if m.role == "user")
    email, category = _email(ticket), _category(ticket)
    lookup = _tool_output(messages, "lookup_customer")
    if lookup is None:
        return FakeProvider.tool_call("lookup_customer", "c1", email=email)
    plan = "unknown" if lookup.is_error else json.loads(lookup.content)["plan"]
    notes = _tool_output(messages, "handoff_to_billing")
    if category == "billing" and notes is None and not lookup.is_error:
        task = f"Customer {email} ({plan} plan) wrote: {_subject(ticket)}"
        return FakeProvider.tool_call("handoff_to_billing", "c2", task=task)
    created = _tool_output(messages, "create_ticket")
    if created is None:
        urgent = plan == "enterprise" or any(w in ticket.lower() for w in ("down", "outage"))
        priority = "P1" if urgent else "P2" if plan in ("pro", "enterprise") else "P3"
        summary = _subject(ticket) + (" (unverified sender)" if lookup.is_error else "")
        return FakeProvider.tool_call(
            "create_ticket",
            "c3",
            category=category,
            priority=priority,
            summary=summary,
            email=email,
        )
    if created.is_error:
        return f"Could not file the ticket: {created.content}"
    filed = next(
        c.arguments for m in reversed(messages) for c in m.tool_calls if c.name == "create_ticket"
    )
    reply = (
        f"Filed {created.content} (category={filed['category']}, "
        f"priority={filed['priority']}, plan={plan})."
    )
    if notes is not None:
        reply += f" Billing specialist: {notes.content}"
    if lookup.is_error:
        reply += " Sender is not a known customer; marked as unverified."
    return reply


def billing_policy(messages: list[Message], tools: list[ToolSpec]) -> Message | str:
    task = next(m.content for m in messages if m.role == "user")
    invoices_msg = _tool_output(messages, "get_invoices")
    if invoices_msg is None:
        return FakeProvider.tool_call("get_invoices", "b1", email=_email(task))
    if invoices_msg.is_error:
        return f"Could not read invoices: {invoices_msg.content}"
    invoices: list[dict[str, Any]] = json.loads(invoices_msg.content)
    seen: dict[tuple[str, float], str] = {}
    for inv in invoices:
        key = (inv["date"], inv["amount"])
        if key in seen:
            return (
                f"Duplicate charge: {inv['id']} repeats {seen[key]} "
                f"({inv['date']}, ${inv['amount']:.2f}); recommend refunding ${inv['amount']:.2f}."
            )
        seen[key] = inv["id"]
    return f"No duplicate charges found across {len(invoices)} invoices."


def fake_provider() -> FakeProvider:
    return FakeProvider(triage_policy, model="fake-triage")

"""LangGraph wiring for Task B.

    persona_builder -> nigerian_context -> domain_resolver -> domain_validator
        |-- (invalid) --> clarification --> retrieval
        |-- (valid)   --> retrieval
    retrieval -> reasoning_ranker -> END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from task_b.agent.clarification_node import clarification_node
from task_b.agent.domain_resolver import domain_resolver_node
from task_b.agent.domain_validator import domain_validator_node
from task_b.agent.nigerian_context import nigerian_context_node
from task_b.agent.persona_builder import persona_builder_node
from task_b.agent.reasoning_ranker import reasoning_ranker_node
from task_b.agent.retrieval_node import retrieval_node
from task_b.agent.state import AgentState


def route_after_validation(state: AgentState) -> str:
    """Conditional edge: 'retrieve' if domain is valid, else 'clarify'."""
    return "retrieve" if state.get("domain_valid") else "clarify"


def build_graph():
    """Construct and compile the Task B LangGraph."""
    g = StateGraph(AgentState)
    g.add_node("persona_builder", persona_builder_node)
    g.add_node("nigerian_context", nigerian_context_node)
    g.add_node("domain_resolver", domain_resolver_node)
    g.add_node("domain_validator", domain_validator_node)
    g.add_node("clarification", clarification_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("reasoning_ranker", reasoning_ranker_node)

    g.set_entry_point("persona_builder")
    g.add_edge("persona_builder", "nigerian_context")
    g.add_edge("nigerian_context", "domain_resolver")
    g.add_edge("domain_resolver", "domain_validator")
    g.add_conditional_edges(
        "domain_validator",
        route_after_validation,
        {"clarify": "clarification", "retrieve": "retrieval"},
    )
    g.add_edge("clarification", "retrieval")
    g.add_edge("retrieval", "reasoning_ranker")
    g.add_edge("reasoning_ranker", END)

    return g.compile()

"""LangGraph wiring for Task A.

Linear pipeline:
    persona_builder -> nigerian_context -> history_grounding -> review_generator -> quality_checker
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from task_a.agent.history_grounding_node import history_grounding_node
from task_a.agent.nigerian_context import nigerian_context_node
from task_a.agent.persona_builder import persona_builder_node
from task_a.agent.quality_checker import quality_checker_node
from task_a.agent.review_generator import review_generator_node
from task_a.agent.state import AgentState


def build_graph():
    """Construct and compile the Task A LangGraph."""
    g = StateGraph(AgentState)
    g.add_node("persona_builder", persona_builder_node)
    g.add_node("nigerian_context", nigerian_context_node)
    g.add_node("history_grounding", history_grounding_node)
    g.add_node("review_generator", review_generator_node)
    g.add_node("quality_checker", quality_checker_node)

    g.set_entry_point("persona_builder")
    g.add_edge("persona_builder", "nigerian_context")
    g.add_edge("nigerian_context", "history_grounding")
    g.add_edge("history_grounding", "review_generator")
    g.add_edge("review_generator", "quality_checker")
    g.add_edge("quality_checker", END)

    return g.compile()

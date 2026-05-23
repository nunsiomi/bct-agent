"""LangGraph wiring for Task A.

Pipeline with self-critique loop:

    persona_builder -> nigerian_context -> history_grounding -> review_generator
        -> critique --(passes OR budget spent)--> quality_checker -> END
                  \\--(fails + budget left)----> revise --> critique (back-edge)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from task_a.agent.critique_node import critique_node, should_revise
from task_a.agent.history_grounding_node import history_grounding_node
from task_a.agent.nigerian_context import nigerian_context_node
from task_a.agent.persona_builder import persona_builder_node
from task_a.agent.quality_checker import quality_checker_node
from task_a.agent.review_generator import review_generator_node
from task_a.agent.revise_node import revise_review_node
from task_a.agent.state import AgentState


def build_graph():
    """Construct and compile the Task A LangGraph."""
    g = StateGraph(AgentState)
    # `persona_builder_node` and `nigerian_context_node` live in `core/` and
    # take a generic `dict[str, Any]` so they can be shared with Task B. They
    # are runtime-compatible with `AgentState` (a TypedDict, i.e. a dict at
    # runtime), but pyright can't narrow that -- the type-ignore is local and
    # intentional.
    g.add_node("persona_builder", persona_builder_node)  # type: ignore[arg-type]
    g.add_node("nigerian_context", nigerian_context_node)  # type: ignore[arg-type]
    g.add_node("history_grounding", history_grounding_node)
    g.add_node("review_generator", review_generator_node)
    g.add_node("critique", critique_node)
    g.add_node("revise", revise_review_node)
    g.add_node("quality_checker", quality_checker_node)

    g.set_entry_point("persona_builder")
    g.add_edge("persona_builder", "nigerian_context")
    g.add_edge("nigerian_context", "history_grounding")
    g.add_edge("history_grounding", "review_generator")
    g.add_edge("review_generator", "critique")
    g.add_conditional_edges(
        "critique",
        should_revise,
        {"revise": "revise", "finalize": "quality_checker"},
    )
    g.add_edge("revise", "critique")
    g.add_edge("quality_checker", END)

    return g.compile()

"""Persona-builder node — parses a free-text persona into a structured fingerprint.

Mirrors `task_b/agent/persona_builder.py`. The body is identical; only the
state import differs so each container stays self-contained.
"""

from __future__ import annotations

from typing import Any

from shared.utils import (
    build_prompt_context,
    call_claude,
    find_similar_users,
    get_anthropic_client,
    load_persona_signals,
    parse_json_block,
    warn,
)
from task_a.agent.state import AgentState


_FINGERPRINT_SYSTEM = (
    "You are a behavioral analyst. Extract a structured persona fingerprint "
    "from the user's free-text description. Respond with JSON only, no preamble, "
    "no markdown fences, no commentary."
)

_FINGERPRINT_USER_TEMPLATE = """Persona description:
{persona}

Return a JSON object with EXACTLY these keys:
{{
  "rating_bias": float in [-2.0, 2.0],   // tendency to rate above (+) or below (-) the mean
  "tone": one of ["formal","balanced","casual","pidgin","detailed","terse"],
  "price_sensitivity": one of ["low","medium","high"],
  "category_affinity": list of short lowercase tags (e.g. ["food","nigerian","restaurants"]),
  "nigerian_markers": list of cultural cues detected in the persona text (e.g. ["lagos","pidgin","jollof"])
}}

JSON only.
"""


def build_persona_profile(persona_text: str) -> dict[str, Any]:
    """Call Claude to extract a structured persona fingerprint from free text."""
    client = get_anthropic_client()
    raw = call_claude(
        client,
        system=_FINGERPRINT_SYSTEM,
        user=_FINGERPRINT_USER_TEMPLATE.format(persona=persona_text),
        max_tokens=500,
        temperature=0.2,
    )
    try:
        parsed = parse_json_block(raw)
    except ValueError as exc:
        warn(f"persona_builder: JSON parse failed ({exc}); using neutral fingerprint")
        parsed = {}

    fingerprint = {
        "rating_bias": float(parsed.get("rating_bias", 0.0) or 0.0),
        "tone": str(parsed.get("tone", "balanced") or "balanced").lower(),
        "price_sensitivity": str(parsed.get("price_sensitivity", "medium") or "medium").lower(),
        "category_affinity": [str(x).lower() for x in (parsed.get("category_affinity") or [])],
        "nigerian_markers": [str(x).lower() for x in (parsed.get("nigerian_markers") or [])],
    }
    return fingerprint


def persona_builder_node(state: AgentState) -> AgentState:
    """Read `state['persona']`; write fingerprint + similar-user cohort to state."""
    persona_text = state.get("persona", "") or ""
    fingerprint = build_persona_profile(persona_text)

    df = load_persona_signals()
    similar = find_similar_users(fingerprint, df, n=5)
    summary = build_prompt_context(similar)
    cohort_pct_5star = float(similar["pct_5star"].mean()) if "pct_5star" in similar else 0.0

    drop_cols = [c for c in similar.columns if c.startswith("_")]
    safe_similar = similar.drop(columns=drop_cols, errors="ignore").to_dict("records")

    state["fingerprint"] = fingerprint
    state["similar_users"] = safe_similar
    state["similar_users_summary"] = summary
    state["cohort_pct_5star"] = cohort_pct_5star
    return state

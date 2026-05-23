"""Reasoning ranker — re-ranks retrieved candidates with persona-aware reasons."""

from __future__ import annotations

import json
from typing import Any

from core.json_utils import parse_json_block, warn
from core.llm import call_claude, get_anthropic_client
from task_b.agent.state import AgentState


_SYSTEM = (
    "You are a Nigerian-aware recommender. Re-rank the candidates against the "
    "persona. Pick the best 5 in priority order and explain each pick in the "
    "persona's terms. If a regional register is provided (Yoruba / Igbo / "
    "Hausa), weave in at most 2 region-specific items across ALL 5 reasons "
    "combined — never per-reason. Return JSON only, no preamble, no markdown."
)


def _regional_block(ctx: dict[str, Any], fingerprint: dict[str, Any]) -> str:
    """Return the regional-palette prompt fragment, or '' when gated off."""
    if not ctx or not ctx.get("apply"):
        return ""
    tone = str(fingerprint.get("tone", "balanced") or "balanced").lower()
    if tone == "formal":
        return ""
    region = ctx.get("language_region", "pidgin_only")
    palette = ctx.get("regional_palette") or {}
    foods = ", ".join(palette.get("food_nouns", []))
    interjections = ", ".join(palette.get("interjections", []))
    praise = palette.get("praise", "")
    disappointment = palette.get("disappointment", "")
    return (
        f"Detected regional register: {region}\n"
        f"  Food/dishes (only if topically relevant): {foods or '—'}\n"
        f"  Interjections (use at most 1): {interjections or '—'}\n"
        f"  Praise phrase: {praise}\n"
        f"  Disappointment phrase: {disappointment}\n"
        "Inject AT MOST 2 region-specific items across all 5 reasons "
        "COMBINED — not per reason. If the candidate doesn't warrant it, "
        "use Pidgin/Nigerian English instead. Never force a phrase."
    )


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        cats = ", ".join(c.get("categories") or [])
        niches = ", ".join(c.get("niches") or [])
        tags = ", ".join(c.get("tags") or [])
        lines.append(
            f"{i}. {c.get('title', '?')} | categories=[{cats}] | niches=[{niches}] | tags=[{tags}]"
        )
    return "\n".join(lines)


def build_ranker_prompt(state: AgentState) -> tuple[str, str]:
    """Compose (system_prompt, user_prompt) for the ranking LLM call."""
    persona = state.get("persona", "")
    fingerprint = state.get("fingerprint", {})
    ctx = state.get("nigerian_context", {}) or {}
    domain = state.get("resolved_domain", "")
    niche = state.get("resolved_niche") or "(none specified)"
    candidates = state.get("candidates", []) or []

    ctx_lines = (
        f"Language register: {ctx.get('language_register', 'neutral')}\n"
        f"Pidgin markers: {', '.join(ctx.get('pidgin_markers', []))}\n"
        f"Cultural vocab: {', '.join(ctx.get('cultural_vocab', []))}\n"
        f"Currency: {ctx.get('currency_symbol', '₦')}"
    )

    regional_block = _regional_block(ctx, fingerprint)
    regional_section = f"\nRegional palette:\n{regional_block}\n" if regional_block else ""

    user = (
        f"Persona description:\n{persona}\n\n"
        f"Persona fingerprint (JSON):\n{json.dumps(fingerprint, ensure_ascii=False)}\n\n"
        f"Nigerian context:\n{ctx_lines}\n"
        f"{regional_section}\n"
        f"Domain: {domain}\nNiche: {niche}\n\n"
        f"Candidates to re-rank (do not invent new items):\n{_format_candidates(candidates)}\n\n"
        "Return JSON ONLY: a list of EXACTLY 5 items, each with keys:\n"
        '  {"rank": int 1-5, "title": str (must match a candidate exactly), '
        '"reason": str (<=140 chars, persona-specific), '
        '"match_score": int in [0, 100]}'
    )
    return _SYSTEM, user


def parse_ranker_output(text: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse Claude's JSON output, drop invalid items, normalise."""
    try:
        parsed = parse_json_block(text)
    except ValueError as exc:
        warn(f"reasoning_ranker: JSON parse failed ({exc}); raw={text[:200]!r}")
        return []

    if isinstance(parsed, dict):
        for k in ("recommendations", "items", "results", "data"):
            if isinstance(parsed.get(k), list):
                parsed = parsed[k]
                break
    if not isinstance(parsed, list):
        return []

    titles = {c.get("title") for c in candidates}
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if title not in titles:
            continue
        try:
            score = float(item.get("match_score", 50.0))
        except (TypeError, ValueError):
            score = 50.0
        if 0.0 <= score <= 1.0 and score != 0 and score != 1:
            score = score * 100.0
        score = max(0.0, min(100.0, score))
        reason = str(item.get("reason", "")).strip()[:140] or "Strong match for the persona."
        out.append({"title": title, "reason": reason, "match_score": score})
        if len(out) >= 5:
            break
    return out


def reasoning_ranker_node(state: AgentState) -> AgentState:
    """Write the final `recommendations` list to state."""
    candidates = state.get("candidates", []) or []
    system, user = build_ranker_prompt(state)

    client = get_anthropic_client()
    raw = call_claude(client, system=system, user=user, max_tokens=900, temperature=0.3)
    items = parse_ranker_output(raw, candidates)

    if len(items) < 5:
        seen = {x["title"] for x in items}
        for c in candidates:
            t = c.get("title")
            if t and t not in seen:
                items.append({
                    "title": t,
                    "reason": "High retrieval score for the persona's domain and niche.",
                    "match_score": 50.0,
                })
                seen.add(t)
                if len(items) >= 5:
                    break

    items = items[:5]
    recommendations = [
        {
            "rank": i + 1,
            "title": x["title"],
            "reason": x["reason"],
            "match_score": round(float(x["match_score"]), 1),
        }
        for i, x in enumerate(items)
    ]
    state["recommendations"] = recommendations
    return state

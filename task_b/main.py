"""Task B entrypoint — FastAPI app exposing `POST /recommend`."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from datasets.utils import load_persona_signals
from datasets.validation import is_gibberish, is_keyboard_mash, is_too_short
from task_b.agent.graph import build_graph


app = FastAPI(title="BCT Task B — Persona-Driven Domain Recommendation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    persona: str
    domain: str
    niche: str | None = None


class Recommendation(BaseModel):
    rank: int
    title: str
    reason: str
    match_score: float


class RecommendResponse(BaseModel):
    recommendations: list[Recommendation]
    fallback_used: bool


load_persona_signals()
_graph = build_graph()


_PERSONA_SUGGESTION = (
    "Lagos professional who loves Nollywood and thrillers"
)


def validate_input(persona: str) -> dict | None:
    """Return None when valid, or a 422-body dict when invalid.

    Persona only — domain handling is owned by domain_validator inside the graph.
    """
    persona = persona or ""

    if is_too_short(persona, 10):
        return {
            "error": "invalid_input",
            "code": "too_short",
            "message": (
                "Your persona description is too short. Please describe the "
                "user in more detail — their location, preferences, and how "
                "they typically write reviews."
            ),
            "field": "persona",
            "suggestion": _PERSONA_SUGGESTION,
        }
    if is_keyboard_mash(persona):
        return {
            "error": "invalid_input",
            "code": "keyboard_mash",
            "message": (
                "We couldn't understand this input. Did you mean to describe "
                "a user persona? Try something like: Lagos professional, "
                "loves action movies, rates generously"
            ),
            "field": "persona",
            "suggestion": _PERSONA_SUGGESTION,
        }
    if is_gibberish(persona):
        return {
            "error": "invalid_input",
            "code": "gibberish",
            "message": (
                "This doesn't look like a valid persona description. Please "
                "describe a real user — for example: Lagos food blogger, "
                "values price, writes in casual Nigerian English"
            ),
            "field": "persona",
            "suggestion": _PERSONA_SUGGESTION,
        }
    return None


def get_graph():
    """Return the compiled LangGraph (built once at module load)."""
    return _graph


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Run the Task B graph end-to-end and return ranked recommendations."""
    err = validate_input(req.persona)
    if err:
        return JSONResponse(status_code=422, content=err)

    try:
        result = _graph.invoke({
            "persona": req.persona,
            "domain": req.domain,
            "niche": req.niche,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent error: {exc}") from exc

    recs = [
        Recommendation(
            rank=int(r.get("rank", i + 1)),
            title=str(r.get("title", "")),
            reason=str(r.get("reason", "")),
            match_score=float(r.get("match_score", 0.0)),
        )
        for i, r in enumerate(result.get("recommendations", []) or [])
    ]
    return RecommendResponse(
        recommendations=recs,
        fallback_used=bool(result.get("fallback_used", False)),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}

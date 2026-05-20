"""Task B entrypoint — FastAPI app exposing `POST /recommend`."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from shared.utils import load_persona_signals
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


def get_graph():
    """Return the compiled LangGraph (built once at module load)."""
    return _graph


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Run the Task B graph end-to-end and return ranked recommendations."""
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

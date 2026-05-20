"""Task A entrypoint — FastAPI app exposing `POST /generate-review`."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from shared.utils import load_persona_signals
from task_a.agent.graph import build_graph


app = FastAPI(title="BCT Task A — Persona-Driven Review Generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    persona: str
    product: str


class ReviewResponse(BaseModel):
    rating: float
    review: str
    confidence: float
    nigerian_context_applied: bool


load_persona_signals()
_graph = build_graph()


def get_graph():
    """Return the compiled LangGraph (built once at module load)."""
    return _graph


@app.post("/generate-review", response_model=ReviewResponse)
def generate_review(req: ReviewRequest) -> ReviewResponse:
    """Run the Task A graph end-to-end and return a structured review."""
    try:
        result = _graph.invoke({"persona": req.persona, "product": req.product})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent error: {exc}") from exc

    return ReviewResponse(
        rating=float(result.get("rating", 3.0)),
        review=str(result.get("review", "")),
        confidence=float(result.get("confidence", 0.0)),
        nigerian_context_applied=bool(result.get("nigerian_context_applied", False)),
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}

"""Task A entrypoint — FastAPI app exposing `POST /generate-review`."""

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


_PERSONA_SUGGESTION = (
    "Lagos food blogger, values price, writes in casual Nigerian English"
)
_PRODUCT_SUGGESTION = "Jollof rice at Chicken Republic Lekki, ₦2500"


def validate_input(persona: str, product: str) -> dict | None:
    """Return None when valid, or a 422-body dict when invalid."""
    persona = persona or ""
    product = product or ""

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
    if is_too_short(product, 5):
        return {
            "error": "invalid_input",
            "code": "too_short",
            "message": (
                "Please describe the product or item in more detail — for "
                "example: Jollof rice at Chicken Republic Lekki, ₦2500"
            ),
            "field": "product",
            "suggestion": _PRODUCT_SUGGESTION,
        }
    if is_keyboard_mash(product) or is_gibberish(product):
        return {
            "error": "invalid_input",
            "code": "gibberish",
            "message": (
                "We couldn't understand the product. Please retype using real "
                "words — for example: Jollof rice at Chicken Republic Lekki, ₦2500"
            ),
            "field": "product",
            "suggestion": _PRODUCT_SUGGESTION,
        }
    return None


def get_graph():
    """Return the compiled LangGraph (built once at module load)."""
    return _graph


@app.post("/generate-review", response_model=ReviewResponse)
def generate_review(req: ReviewRequest):
    """Run the Task A graph end-to-end and return a structured review."""
    err = validate_input(req.persona, req.product)
    if err:
        return JSONResponse(status_code=422, content=err)

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

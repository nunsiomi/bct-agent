"""Typed I/O contracts shared across the system.

Use these models at the API boundary and inside nodes that emit structured
data. Untyped `dict[str, Any]` is allowed inside graph state (LangGraph's
TypedDict pattern) but every external response should round-trip through a
Pydantic model here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, confloat


Tone = Literal["formal", "balanced", "casual", "pidgin", "detailed", "terse"]
PriceSens = Literal["low", "medium", "high"]
LanguageRegion = Literal["yoruba", "igbo", "hausa", "pidgin_only"]


class PersonaFingerprint(BaseModel):
    rating_bias: float = Field(0.0, ge=-2.0, le=2.0)
    tone: Tone = "balanced"
    price_sensitivity: PriceSens = "medium"
    category_affinity: list[str] = Field(default_factory=list)
    nigerian_markers: list[str] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    rating: confloat(ge=1.0, le=5.0)
    review: str = Field(min_length=20, max_length=1000)
    confidence: confloat(ge=0.0, le=1.0) = 0.0
    nigerian_context_applied: bool = False


class Recommendation(BaseModel):
    rank: int = Field(ge=1)
    title: str
    reason: str
    match_score: confloat(ge=0.0, le=1.0)


class RecommendationList(BaseModel):
    recommendations: list[Recommendation]
    fallback_used: bool = False
    clarification_question: Optional[str] = None

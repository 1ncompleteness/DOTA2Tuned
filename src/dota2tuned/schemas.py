from __future__ import annotations

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    hero_id: int
    hero_name: str
    role: str | None = None
    score: float
    win_prob_delta: float | None = None
    counter_lift: float | None = None
    synergy_lift: float | None = None
    sample_size: int
    patch: str
    scope: str
    sources: list[str] = Field(default_factory=list)
    confidence: str
    caveats: list[str] = Field(default_factory=list)


class DraftInput(BaseModel):
    allied_heroes: list[int] = Field(default_factory=list)
    enemy_heroes: list[int] = Field(default_factory=list)
    banned_heroes: list[int] = Field(default_factory=list)
    role: str | None = None
    rank_bracket: str = "pro"
    patch: str = "current"
    scope: str = "pro"

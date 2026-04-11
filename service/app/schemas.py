from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    text: str
    line_order: int | None = None
    ocr_confidence: float | None = None
    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None


class LineRolePrediction(BaseModel):
    text: str
    predicted_label: str
    predicted_score: float | None = None
    line_order: int | None = None
    ocr_confidence: float | None = None
    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None


class ParsedItem(BaseModel):
    local_id: str
    dish_name: str
    description: str | None = None
    section: str | None = None
    price_value: float | None = None
    price_currency: str | None = None
    price_text: str | None = None
    explicit_allergens: list[str] = Field(default_factory=list)
    parser_confidence: float | None = None


class ParseTextRequest(BaseModel):
    lines: list[str | OCRLine]


class ParseResponse(BaseModel):
    n_lines: int
    n_items: int
    parser_module: str | None = None
    line_role_model_loaded: bool = False
    ocr_lines: list[OCRLine] = Field(default_factory=list)
    line_roles: list[LineRolePrediction] = Field(default_factory=list)
    items: list[ParsedItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    api_title: str
    api_version: str


class DatasetStatsResponse(BaseModel):
    summary: dict[str, Any]


class ParserStatsResponse(BaseModel):
    metrics: dict[str, Any]
    active_parser_module: str | None = None
    active_metrics_dir: str | None = None
    comparison_rows: list[dict[str, Any]] = Field(default_factory=list)


class LineRoleStatsResponse(BaseModel):
    metrics: dict[str, Any]
    report: dict[str, Any] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    items: list[ParsedItem]
    craving_text: str | None = None
    liked_terms: list[str] = Field(default_factory=list)
    disliked_terms: list[str] = Field(default_factory=list)
    excluded_allergens: list[str] = Field(default_factory=list)
    preferred_sections: list[str] = Field(default_factory=list)
    excluded_sections: list[str] = Field(default_factory=list)
    max_price: float | None = None
    top_k: int = 5
    engine: Literal["auto", "tfidf", "sentence_transformer"] = "auto"


class RecommendationRow(BaseModel):
    rank: int
    local_id: str
    dish_name: str
    section: str | None = None
    price_value: float | None = None
    score: float
    semantic_score: float
    rule_score: float
    reasons: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    engine_used: str
    n_candidates: int
    rows: list[RecommendationRow] = Field(default_factory=list)

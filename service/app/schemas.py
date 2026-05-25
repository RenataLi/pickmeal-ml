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
    ingredient_hints: list[str] = Field(default_factory=list)
    explicit_allergens: list[str] = Field(default_factory=list)
    diet_flags: dict[str, bool] = Field(default_factory=dict)
    calories_low: float | None = None
    calories_mid: float | None = None
    calories_high: float | None = None
    nutrition_confidence: float | None = None
    enrichment_notes: list[str] = Field(default_factory=list)
    llm_summary: str | None = None
    llm_why_it_fits: str | None = None
    llm_caution_note: str | None = None
    parser_confidence: float | None = None


class ParseTextRequest(BaseModel):
    lines: list[str | OCRLine]


class ParseLinesRequest(BaseModel):
    ocr_lines: list[OCRLine]
    ocr_backend: str | None = None


class ParseResponse(BaseModel):
    n_lines: int
    n_items: int
    session_id: str | None = None
    ocr_backend: str | None = None
    requested_ocr_backend: str | None = None
    ocr_backend_warning: str | None = None
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


class OCRStatsResponse(BaseModel):
    requested_backend: str
    fallback_backend: str | None = None
    available_backends: dict[str, bool] = Field(default_factory=dict)
    paddle_cache_dir: str | None = None
    paddle_detection_model: str | None = None
    paddle_mobile_detection_model: str | None = None
    ocr_max_image_side: int | None = None


class OCRImageResponse(BaseModel):
    backend: str
    requested_backend: str | None = None
    warning: str | None = None
    n_lines: int
    lines: list[OCRLine] = Field(default_factory=list)


class StorageStatsResponse(BaseModel):
    enabled: bool
    initialized: bool
    database_url_present: bool
    embedding_model_name: str | None = None
    embedding_dimensions: int | None = None
    vector_backend: str | None = None
    row_counts: dict[str, int] = Field(default_factory=dict)
    source_kind_counts: dict[str, int] = Field(default_factory=dict)
    last_error: str | None = None


class LLMStatsResponse(BaseModel):
    enabled: bool
    configured: bool
    base_url_present: bool
    api_key_present: bool
    model: str | None = None
    timeout_seconds: int | None = None
    max_items_per_request: int | None = None
    last_error: str | None = None


class RAGEvidenceRow(BaseModel):
    item_local_id: str
    source_type: str
    source_id: str
    title: str
    similarity: float
    content_preview: str


class RAGStatsResponse(BaseModel):
    enabled: bool
    initialized: bool
    vector_backend: str | None = None
    top_k: int | None = None
    dataset_path: str | None = None
    document_count: int = 0
    source_type_counts: dict[str, int] = Field(default_factory=dict)
    last_error: str | None = None


class NutritionStatsResponse(BaseModel):
    enabled: bool
    initialized: bool
    database_url_present: bool
    seed_path: str | None = None
    latest_version: dict[str, Any] | None = None
    row_counts: dict[str, int] = Field(default_factory=dict)
    source_counts: dict[str, int] = Field(default_factory=dict)
    last_error: str | None = None


class NutritionRefreshRequest(BaseModel):
    force: bool = False


class NutritionRefreshResponse(BaseModel):
    source_name: str
    source_version: str
    version_id: int
    record_count: int
    skipped: bool = False


class CacheStatsResponse(BaseModel):
    enabled: bool
    configured: bool
    available: bool
    redis_url_present: bool
    namespace: str
    recommendation_ttl_seconds: int
    llm_ttl_seconds: int
    get_count: int = 0
    hit_count: int = 0
    miss_count: int = 0
    set_count: int = 0
    error_count: int = 0
    local_counters: dict[str, int] = Field(default_factory=dict)
    namespace_key_counts: dict[str, int] = Field(default_factory=dict)
    server_info: dict[str, Any] = Field(default_factory=dict)
    last_error: str | None = None


class RuntimeStatsResponse(BaseModel):
    service_id: str
    started_at: str
    uptime_seconds: float
    request_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    path_rows: list[dict[str, Any]] = Field(default_factory=list)
    stage_rows: list[dict[str, Any]] = Field(default_factory=list)
    recent_errors: list[dict[str, Any]] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    session_id: str | None = None
    items: list[ParsedItem]
    craving_text: str | None = None
    liked_terms: list[str] = Field(default_factory=list)
    disliked_terms: list[str] = Field(default_factory=list)
    excluded_allergens: list[str] = Field(default_factory=list)
    required_diet_flags: list[str] = Field(default_factory=list)
    preferred_sections: list[str] = Field(default_factory=list)
    excluded_sections: list[str] = Field(default_factory=list)
    max_price: float | None = None
    max_calories: float | None = None
    combo_budget: float | None = None
    combo_max_calories: float | None = None
    combo_min_items: int = 2
    combo_max_items: int = 3
    top_k: int = 5
    engine: Literal["auto", "tfidf", "sentence_transformer", "catboost_reranker"] = "auto"


class RecommendationRow(BaseModel):
    rank: int
    local_id: str
    dish_name: str
    section: str | None = None
    price_value: float | None = None
    calories_mid: float | None = None
    diet_flags: list[str] = Field(default_factory=list)
    match_label: str | None = None
    score: float
    semantic_score: float
    rule_score: float
    reasons: list[str] = Field(default_factory=list)


class CombinationRow(BaseModel):
    rank: int
    item_ids: list[str] = Field(default_factory=list)
    dish_names: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    total_price: float | None = None
    total_calories: float | None = None
    score: float
    match_label: str | None = None
    reasons: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    recommendation_id: str | None = None
    engine_used: str
    n_candidates: int
    rows: list[RecommendationRow] = Field(default_factory=list)
    combo_rows: list[CombinationRow] = Field(default_factory=list)


class SimilarDishesRequest(BaseModel):
    query_text: str
    top_k: int = 5


class SimilarDishRow(BaseModel):
    session_id: str
    local_id: str
    dish_name: str
    section: str | None = None
    similarity: float


class SimilarDishesResponse(BaseModel):
    rows: list[SimilarDishRow] = Field(default_factory=list)


class StorageSnapshotResponse(BaseModel):
    schema_version: int
    exported_at: str
    vector_backend: str | None = None
    row_counts: dict[str, int] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class StorageSnapshotImportRequest(BaseModel):
    snapshot: dict[str, Any]


class StorageSnapshotImportResponse(BaseModel):
    imported_counts: dict[str, int] = Field(default_factory=dict)
    replaced_existing: bool = True


class LLMItemEnrichmentRequest(BaseModel):
    items: list[ParsedItem]
    user_context: str | None = None
    top_k: int = 5


class LLMItemEnrichmentResponse(BaseModel):
    provider_label: str
    model: str
    n_items_requested: int
    n_items_returned: int
    items: list[ParsedItem] = Field(default_factory=list)
    retrieval_rows: list[RAGEvidenceRow] = Field(default_factory=list)

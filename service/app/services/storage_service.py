from __future__ import annotations

import json
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..config import get_settings
from ..schemas import (
    CombinationRow,
    LineRolePrediction,
    OCRLine,
    ParsedItem,
    RecommendRequest,
    RecommendationRow,
)


_RUNTIME_STATE: dict[str, Any] = {
    "enabled": False,
    "initialized": False,
    "last_error": None,
}


def _load_psycopg():
    try:
        import psycopg
    except Exception:
        return None
    return psycopg


def _storage_enabled() -> bool:
    settings = get_settings()
    return bool(settings.database_url)


@lru_cache(maxsize=1)
def _embedding_vectorizer() -> HashingVectorizer:
    settings = get_settings()
    return HashingVectorizer(
        n_features=settings.embedding_dimensions,
        alternate_sign=False,
        analyzer="word",
        ngram_range=(1, 2),
        norm=None,
    )


def _embed_text(text: str) -> list[float]:
    matrix = _embedding_vectorizer().transform([text or ""])
    vector = matrix.toarray().astype(np.float32)[0]
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return [float(x) for x in vector.tolist()]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _item_embedding_text(item: ParsedItem) -> str:
    parts = [item.dish_name or "", item.description or "", item.section or ""]
    return " ".join(part.strip() for part in parts if part).strip()


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def initialize_storage(force: bool = False) -> None:
    settings = get_settings()
    _RUNTIME_STATE["enabled"] = _storage_enabled()

    if not settings.database_url:
        _RUNTIME_STATE["initialized"] = False
        _RUNTIME_STATE["last_error"] = None
        return

    if _RUNTIME_STATE["initialized"] and not force:
        return

    psycopg = _load_psycopg()
    if psycopg is None:
        _RUNTIME_STATE["initialized"] = False
        _RUNTIME_STATE["last_error"] = "psycopg is not installed"
        return

    dims = int(settings.embedding_dimensions)
    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS menu_sessions (
                        session_id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        source_kind TEXT NOT NULL,
                        ocr_backend TEXT,
                        parser_module TEXT,
                        n_lines INTEGER NOT NULL,
                        n_items INTEGER NOT NULL,
                        raw_ocr_lines JSONB NOT NULL DEFAULT '[]'::jsonb,
                        line_roles JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS parsed_items (
                        id BIGSERIAL PRIMARY KEY,
                        session_id UUID NOT NULL REFERENCES menu_sessions(session_id) ON DELETE CASCADE,
                        local_id TEXT NOT NULL,
                        dish_name TEXT NOT NULL,
                        description TEXT,
                        section TEXT,
                        price_value DOUBLE PRECISION,
                        price_currency TEXT,
                        price_text TEXT,
                        ingredient_hints JSONB NOT NULL DEFAULT '[]'::jsonb,
                        explicit_allergens JSONB NOT NULL DEFAULT '[]'::jsonb,
                        diet_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
                        calories_low DOUBLE PRECISION,
                        calories_mid DOUBLE PRECISION,
                        calories_high DOUBLE PRECISION,
                        nutrition_confidence DOUBLE PRECISION,
                        enrichment_notes JSONB NOT NULL DEFAULT '[]'::jsonb,
                        parser_confidence DOUBLE PRECISION
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS dish_embeddings (
                        id BIGSERIAL PRIMARY KEY,
                        session_id UUID NOT NULL REFERENCES menu_sessions(session_id) ON DELETE CASCADE,
                        local_id TEXT NOT NULL,
                        text_value TEXT NOT NULL,
                        embedding_model TEXT NOT NULL,
                        embedding VECTOR({dims}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendation_runs (
                        recommendation_id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        session_id UUID REFERENCES menu_sessions(session_id) ON DELETE SET NULL,
                        request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                        engine_used TEXT NOT NULL,
                        n_candidates INTEGER NOT NULL,
                        recommendation_rows JSONB NOT NULL DEFAULT '[]'::jsonb,
                        combo_rows JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_parsed_items_session_id ON parsed_items (session_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_dish_embeddings_session_id ON dish_embeddings (session_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_recommendation_runs_session_id ON recommendation_runs (session_id)")
        _RUNTIME_STATE["initialized"] = True
        _RUNTIME_STATE["last_error"] = None
    except Exception as exc:
        _RUNTIME_STATE["initialized"] = False
        _RUNTIME_STATE["last_error"] = str(exc)


def persist_parse_result(
    source_kind: str,
    ocr_backend: str | None,
    parser_module: str | None,
    ocr_lines: list[OCRLine],
    line_roles: list[LineRolePrediction],
    items: list[ParsedItem],
) -> str | None:
    initialize_storage()
    settings = get_settings()
    if not settings.database_url or not _RUNTIME_STATE["initialized"]:
        return None

    psycopg = _load_psycopg()
    if psycopg is None:
        return None

    session_id = uuid.uuid4()
    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO menu_sessions (
                        session_id,
                        source_kind,
                        ocr_backend,
                        parser_module,
                        n_lines,
                        n_items,
                        raw_ocr_lines,
                        line_roles
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        session_id,
                        source_kind,
                        ocr_backend,
                        parser_module,
                        len(ocr_lines),
                        len(items),
                        _json_dumps([line.model_dump(mode="json") for line in ocr_lines]),
                        _json_dumps([line_role.model_dump(mode="json") for line_role in line_roles]),
                    ),
                )
                for item in items:
                    cur.execute(
                        """
                        INSERT INTO parsed_items (
                            session_id,
                            local_id,
                            dish_name,
                            description,
                            section,
                            price_value,
                            price_currency,
                            price_text,
                            ingredient_hints,
                            explicit_allergens,
                            diet_flags,
                            calories_low,
                            calories_mid,
                            calories_high,
                            nutrition_confidence,
                            enrichment_notes,
                            parser_confidence
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            session_id,
                            item.local_id,
                            item.dish_name,
                            item.description,
                            item.section,
                            item.price_value,
                            item.price_currency,
                            item.price_text,
                            _json_dumps(item.ingredient_hints),
                            _json_dumps(item.explicit_allergens),
                            _json_dumps(item.diet_flags),
                            item.calories_low,
                            item.calories_mid,
                            item.calories_high,
                            item.nutrition_confidence,
                            _json_dumps(item.enrichment_notes),
                            item.parser_confidence,
                        ),
                    )
                    text_value = _item_embedding_text(item)
                    if not text_value:
                        continue
                    cur.execute(
                        """
                        INSERT INTO dish_embeddings (
                            session_id,
                            local_id,
                            text_value,
                            embedding_model,
                            embedding,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                        """,
                        (
                            session_id,
                            item.local_id,
                            text_value,
                            settings.embedding_model_name,
                            _vector_literal(_embed_text(text_value)),
                            _json_dumps(
                                {
                                    "dish_name": item.dish_name,
                                    "section": item.section,
                                    "price_value": item.price_value,
                                }
                            ),
                        ),
                    )
        _RUNTIME_STATE["last_error"] = None
        return str(session_id)
    except Exception as exc:
        _RUNTIME_STATE["last_error"] = str(exc)
        return None


def persist_recommendation_result(
    request_payload: RecommendRequest,
    engine_used: str,
    n_candidates: int,
    rows: list[RecommendationRow],
    combo_rows: list[CombinationRow],
) -> str | None:
    initialize_storage()
    settings = get_settings()
    if not settings.database_url or not _RUNTIME_STATE["initialized"]:
        return None

    psycopg = _load_psycopg()
    if psycopg is None:
        return None

    recommendation_id = uuid.uuid4()
    try:
        session_id = uuid.UUID(request_payload.session_id) if request_payload.session_id else None
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recommendation_runs (
                        recommendation_id,
                        session_id,
                        request_payload,
                        engine_used,
                        n_candidates,
                        recommendation_rows,
                        combo_rows
                    )
                    VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        recommendation_id,
                        session_id,
                        _json_dumps(request_payload.model_dump(mode="json")),
                        engine_used,
                        n_candidates,
                        _json_dumps([row.model_dump(mode="json") for row in rows]),
                        _json_dumps([row.model_dump(mode="json") for row in combo_rows]),
                    ),
                )
        _RUNTIME_STATE["last_error"] = None
        return str(recommendation_id)
    except Exception as exc:
        _RUNTIME_STATE["last_error"] = str(exc)
        return None


def find_similar_dishes(query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    initialize_storage()
    settings = get_settings()
    if not settings.database_url or not _RUNTIME_STATE["initialized"] or not query_text.strip():
        return []

    psycopg = _load_psycopg()
    if psycopg is None:
        return []

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                query_vector = _vector_literal(_embed_text(query_text))
                cur.execute(
                    """
                    SELECT
                        de.session_id::text,
                        de.local_id,
                        COALESCE(pi.dish_name, de.text_value) AS dish_name,
                        pi.section,
                        1 - (de.embedding <=> %s::vector) AS similarity
                    FROM dish_embeddings AS de
                    LEFT JOIN parsed_items AS pi
                        ON pi.session_id = de.session_id
                       AND pi.local_id = de.local_id
                    ORDER BY de.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, query_vector, max(1, int(top_k))),
                )
                rows = cur.fetchall()
        _RUNTIME_STATE["last_error"] = None
        return [
            {
                "session_id": session_id,
                "local_id": local_id,
                "dish_name": dish_name,
                "section": section,
                "similarity": round(float(similarity), 4),
            }
            for session_id, local_id, dish_name, section, similarity in rows
        ]
    except Exception as exc:
        _RUNTIME_STATE["last_error"] = str(exc)
        return []


def load_storage_stats() -> dict[str, Any]:
    initialize_storage()
    settings = get_settings()
    payload: dict[str, Any] = {
        "enabled": _storage_enabled(),
        "initialized": bool(_RUNTIME_STATE["initialized"]),
        "database_url_present": bool(settings.database_url),
        "embedding_model_name": settings.embedding_model_name,
        "embedding_dimensions": settings.embedding_dimensions,
        "row_counts": {},
        "last_error": _RUNTIME_STATE["last_error"],
    }
    if not settings.database_url or not _RUNTIME_STATE["initialized"]:
        return payload

    psycopg = _load_psycopg()
    if psycopg is None:
        payload["last_error"] = "psycopg is not installed"
        return payload

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                row_counts: dict[str, int] = {}
                for table_name in ["menu_sessions", "parsed_items", "dish_embeddings", "recommendation_runs"]:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_counts[table_name] = int(cur.fetchone()[0])
        payload["row_counts"] = row_counts
        payload["last_error"] = None
    except Exception as exc:
        payload["last_error"] = str(exc)
    return payload

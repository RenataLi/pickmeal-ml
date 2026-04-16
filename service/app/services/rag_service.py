from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_settings
from ..schemas import ParsedItem, RAGEvidenceRow
from .enrichment_service import (
    ALLERGEN_MAP,
    ANIMAL_INGREDIENTS,
    CALORIE_ADJUSTMENTS,
    CATEGORY_RULES,
    GLUTEN_INGREDIENTS,
    INGREDIENT_KEYWORDS,
    MEAT_INGREDIENTS,
    SEAFOOD_INGREDIENTS,
)
from .storage_service import (
    _cosine_similarity,
    _embed_text,
    _json_dumps,
    _load_psycopg,
    _vector_literal,
    initialize_storage,
    load_storage_stats,
)


_RAG_STATE: dict[str, Any] = {
    "enabled": False,
    "initialized": False,
    "document_count": 0,
    "source_type_counts": {},
    "vector_backend": None,
    "last_error": None,
}


def _rag_enabled() -> bool:
    settings = get_settings()
    return bool(settings.rag_enabled and settings.database_url)


def _dataset_path() -> Path:
    settings = get_settings()
    return settings.project_root / settings.rag_dataset_path


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _build_static_rule_documents() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for ingredient, keywords in sorted(INGREDIENT_KEYWORDS.items()):
        allergen = ALLERGEN_MAP.get(ingredient)
        content_parts = [
            f"Ingredient keyword rule for {ingredient}.",
            f"Detect this ingredient when menu text mentions: {', '.join(keywords)}.",
        ]
        if allergen:
            content_parts.append(f"This ingredient implies allergen flag: {allergen}.")
        if ingredient in MEAT_INGREDIENTS:
            content_parts.append("This ingredient is not vegetarian or vegan.")
        if ingredient in SEAFOOD_INGREDIENTS:
            content_parts.append("This ingredient is seafood and not vegetarian or vegan.")
        if ingredient in GLUTEN_INGREDIENTS:
            content_parts.append("This ingredient conflicts with gluten_free.")
        if ingredient == "milk":
            content_parts.append("This ingredient conflicts with dairy_free and vegan.")
        if ingredient == "egg":
            content_parts.append("This ingredient conflicts with vegan.")
        docs.append(
            {
                "doc_key": f"ingredient_rule:{ingredient}",
                "source_type": "ingredient_rule",
                "source_id": ingredient,
                "title": f"Ingredient rule: {ingredient}",
                "content": " ".join(content_parts),
                "metadata": {
                    "ingredient": ingredient,
                    "allergen": allergen,
                    "keywords": keywords,
                },
            }
        )

    docs.append(
        {
            "doc_key": "diet_policy:derived_flags",
            "source_type": "policy_rule",
            "source_id": "derived_diet_flags",
            "title": "Diet flag derivation policy",
            "content": (
                "Vegetarian excludes meat and seafood ingredients. "
                "Vegan excludes all vegetarian conflicts plus milk, egg, and honey. "
                "Gluten_free excludes wheat. "
                "Dairy_free excludes milk."
            ),
            "metadata": {
                "meat_ingredients": sorted(MEAT_INGREDIENTS),
                "seafood_ingredients": sorted(SEAFOOD_INGREDIENTS),
                "animal_ingredients": sorted(ANIMAL_INGREDIENTS),
                "gluten_ingredients": sorted(GLUTEN_INGREDIENTS),
            },
        }
    )

    for keywords, triple in CATEGORY_RULES:
        label = "_".join(keywords)
        docs.append(
            {
                "doc_key": f"calorie_rule:{label}",
                "source_type": "calorie_rule",
                "source_id": label,
                "title": f"Calorie base range: {', '.join(keywords)}",
                "content": (
                    f"Base calorie heuristic for dishes matching keywords {', '.join(keywords)} "
                    f"is low={triple[0]}, mid={triple[1]}, high={triple[2]}."
                ),
                "metadata": {
                    "keywords": list(keywords),
                    "calories_low": triple[0],
                    "calories_mid": triple[1],
                    "calories_high": triple[2],
                },
            }
        )

    for ingredient, triple in sorted(CALORIE_ADJUSTMENTS.items()):
        docs.append(
            {
                "doc_key": f"calorie_adjustment:{ingredient}",
                "source_type": "calorie_adjustment",
                "source_id": ingredient,
                "title": f"Calorie adjustment: {ingredient}",
                "content": (
                    f"When ingredient hint {ingredient} is present, add approximately "
                    f"low={triple[0]}, mid={triple[1]}, high={triple[2]} calories to the base estimate."
                ),
                "metadata": {
                    "ingredient": ingredient,
                    "calories_low": triple[0],
                    "calories_mid": triple[1],
                    "calories_high": triple[2],
                },
            }
        )

    return docs


def _build_gold_menu_documents() -> list[dict[str, Any]]:
    dataset_path = _dataset_path()
    if not dataset_path.exists():
        return []

    df = pd.read_csv(dataset_path)
    docs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for row in df.to_dict(orient="records"):
        dish_name = _clean_text(row.get("dish_name"))
        if not dish_name:
            continue
        source_id = _clean_text(row.get("item_id")) or f"{_clean_text(row.get('menu_id'))}:{dish_name}"
        doc_key = f"annotated_menu_item:{source_id}"
        if doc_key in seen_keys:
            continue
        seen_keys.add(doc_key)

        section = _clean_text(row.get("section"))
        description = _clean_text(row.get("description"))
        ingredients = _parse_list(row.get("explicit_ingredients"))
        allergens = _parse_list(row.get("explicit_allergens"))
        price_text = _clean_text(row.get("price_text"))
        qa_note = _clean_text(row.get("qa_note"))
        review_reason = _clean_text(row.get("review_reason"))
        confidence = row.get("confidence")

        content_parts = [
            "Annotated menu dish example.",
            f"Dish name: {dish_name}.",
        ]
        if section:
            content_parts.append(f"Section: {section}.")
        if description:
            content_parts.append(f"Description: {description}.")
        if ingredients:
            content_parts.append(f"Explicit ingredients: {', '.join(ingredients)}.")
        if allergens:
            content_parts.append(f"Explicit allergens: {', '.join(allergens)}.")
        if price_text:
            content_parts.append(f"Observed menu price text: {price_text}.")
        if qa_note:
            content_parts.append(f"QA note: {qa_note}.")
        if review_reason:
            content_parts.append(f"Review note: {review_reason}.")

        docs.append(
            {
                "doc_key": doc_key,
                "source_type": "annotated_menu_item",
                "source_id": source_id,
                "title": f"{dish_name} ({section or 'menu item'})",
                "content": " ".join(content_parts),
                "metadata": {
                    "menu_id": _clean_text(row.get("menu_id")),
                    "page_id": _clean_text(row.get("page_id")),
                    "item_id": source_id,
                    "section": section,
                    "dish_name": dish_name,
                    "description": description,
                    "explicit_ingredients": ingredients,
                    "explicit_allergens": allergens,
                    "price_text": price_text,
                    "confidence": float(confidence) if confidence is not None and str(confidence) != "nan" else None,
                    "needs_review": bool(row.get("needs_review")) if row.get("needs_review") is not None else False,
                },
            }
        )

    return docs


def _all_seed_documents() -> list[dict[str, Any]]:
    return _build_static_rule_documents() + _build_gold_menu_documents()


def initialize_rag(force: bool = False) -> None:
    initialize_storage(force=force)
    settings = get_settings()
    _RAG_STATE["enabled"] = _rag_enabled()

    if not _rag_enabled():
        _RAG_STATE["initialized"] = False
        _RAG_STATE["last_error"] = None
        return

    if _RAG_STATE["initialized"] and not force:
        return

    psycopg = _load_psycopg()
    if psycopg is None:
        _RAG_STATE["initialized"] = False
        _RAG_STATE["last_error"] = "psycopg is not installed"
        return

    dims = int(settings.embedding_dimensions)
    storage_stats = load_storage_stats()
    vector_backend = storage_stats.get("vector_backend")

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        id BIGSERIAL PRIMARY KEY,
                        doc_key TEXT NOT NULL UNIQUE,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        embedding_model TEXT NOT NULL,
                        embedding_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                if vector_backend == "pgvector":
                    cur.execute(f"ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS embedding VECTOR({dims})")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_documents_source_type ON rag_documents (source_type)")

                if force:
                    cur.execute("DELETE FROM rag_documents")

                cur.execute("SELECT COUNT(*) FROM rag_documents")
                current_count = int(cur.fetchone()[0])

                if current_count == 0:
                    docs = _all_seed_documents()
                    for doc in docs:
                        embedding = _embed_text(f"{doc['title']} {doc['content']}")
                        cur.execute(
                            """
                            INSERT INTO rag_documents (
                                doc_key,
                                source_type,
                                source_id,
                                title,
                                content,
                                metadata,
                                embedding_model,
                                embedding_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                            ON CONFLICT (doc_key) DO UPDATE SET
                                source_type = EXCLUDED.source_type,
                                source_id = EXCLUDED.source_id,
                                title = EXCLUDED.title,
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                embedding_model = EXCLUDED.embedding_model,
                                embedding_json = EXCLUDED.embedding_json
                            """,
                            (
                                doc["doc_key"],
                                doc["source_type"],
                                doc["source_id"],
                                doc["title"],
                                doc["content"],
                                _json_dumps(doc["metadata"]),
                                settings.embedding_model_name,
                                _json_dumps(embedding),
                            ),
                        )
                        if vector_backend == "pgvector":
                            cur.execute(
                                """
                                UPDATE rag_documents
                                SET embedding = %s::vector
                                WHERE doc_key = %s
                                """,
                                (_vector_literal(embedding), doc["doc_key"]),
                            )

                cur.execute("SELECT COUNT(*) FROM rag_documents")
                _RAG_STATE["document_count"] = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT source_type, COUNT(*)
                    FROM rag_documents
                    GROUP BY source_type
                    ORDER BY source_type
                    """
                )
                _RAG_STATE["source_type_counts"] = {str(source_type): int(count) for source_type, count in cur.fetchall()}

        _RAG_STATE["initialized"] = True
        _RAG_STATE["vector_backend"] = vector_backend
        _RAG_STATE["last_error"] = None
    except Exception as exc:
        _RAG_STATE["initialized"] = False
        _RAG_STATE["document_count"] = 0
        _RAG_STATE["source_type_counts"] = {}
        _RAG_STATE["vector_backend"] = vector_backend
        _RAG_STATE["last_error"] = str(exc)


def _item_query_text(item: ParsedItem, user_context: str | None = None) -> str:
    parts = [
        item.dish_name or "",
        item.dish_name or "",
        item.section or "",
        item.description or "",
        "ingredients: " + ", ".join(item.ingredient_hints) if item.ingredient_hints else "",
        "allergens: " + ", ".join(item.explicit_allergens) if item.explicit_allergens else "",
        "diet flags: " + ", ".join(flag for flag, enabled in item.diet_flags.items() if enabled) if item.diet_flags else "",
    ]
    return " ".join(part.strip() for part in parts if part).strip()


def retrieve_rag_evidence(item: ParsedItem, user_context: str | None = None, top_k: int | None = None) -> list[RAGEvidenceRow]:
    initialize_rag()
    settings = get_settings()
    if not _rag_enabled() or not _RAG_STATE["initialized"]:
        return []

    query_text = _item_query_text(item, user_context=user_context)
    if not query_text:
        return []

    psycopg = _load_psycopg()
    if psycopg is None:
        return []

    requested_top_k = max(1, min(int(top_k or settings.rag_top_k), 6))
    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                if _RAG_STATE.get("vector_backend") == "pgvector":
                    query_vector = _vector_literal(_embed_text(query_text))
                    cur.execute(
                        """
                        SELECT
                            source_type,
                            source_id,
                            title,
                            content,
                            1 - (embedding <=> %s::vector) AS similarity
                        FROM rag_documents
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_vector, query_vector, requested_top_k),
                    )
                    rows = cur.fetchall()
                else:
                    cur.execute(
                        """
                        SELECT
                            source_type,
                            source_id,
                            title,
                            content,
                            embedding_json
                        FROM rag_documents
                        """
                    )
                    query_embedding = _embed_text(query_text)
                    scored_rows = []
                    for source_type, source_id, title, content, embedding_json in cur.fetchall():
                        similarity = _cosine_similarity(query_embedding, embedding_json or [])
                        scored_rows.append((source_type, source_id, title, content, similarity))
                    scored_rows.sort(key=lambda row: row[4], reverse=True)
                    rows = scored_rows[:requested_top_k]

        _RAG_STATE["last_error"] = None
        return [
            RAGEvidenceRow(
                item_local_id=item.local_id,
                source_type=str(source_type),
                source_id=str(source_id),
                title=str(title),
                similarity=round(float(similarity), 4),
                content_preview=str(content)[:280],
            )
            for source_type, source_id, title, content, similarity in rows
        ]
    except Exception as exc:
        _RAG_STATE["last_error"] = str(exc)
        return []


def retrieve_rag_for_items(items: list[ParsedItem], user_context: str | None = None, top_k: int | None = None) -> tuple[dict[str, list[RAGEvidenceRow]], list[RAGEvidenceRow]]:
    evidence_by_item: dict[str, list[RAGEvidenceRow]] = {}
    flat_rows: list[RAGEvidenceRow] = []
    for item in items:
        evidence_rows = retrieve_rag_evidence(item, user_context=user_context, top_k=top_k)
        evidence_by_item[item.local_id] = evidence_rows
        flat_rows.extend(evidence_rows)
    return evidence_by_item, flat_rows


def load_rag_stats() -> dict[str, Any]:
    initialize_rag()
    settings = get_settings()
    return {
        "enabled": _rag_enabled(),
        "initialized": bool(_RAG_STATE["initialized"]),
        "vector_backend": _RAG_STATE["vector_backend"],
        "top_k": settings.rag_top_k,
        "dataset_path": str(_dataset_path()),
        "document_count": int(_RAG_STATE["document_count"]),
        "source_type_counts": dict(_RAG_STATE["source_type_counts"]),
        "last_error": _RAG_STATE["last_error"],
    }

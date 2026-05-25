from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings


_NUTRITION_STATE: dict[str, Any] = {
    "enabled": False,
    "initialized": False,
    "last_error": None,
    "latest_version": None,
}


def _load_psycopg():
    try:
        import psycopg
    except Exception:
        return None
    return psycopg


def _nutrition_enabled() -> bool:
    settings = get_settings()
    return bool(settings.database_url and settings.nutrition_reference_enabled)


def _seed_path() -> Path:
    settings = get_settings()
    return settings.project_root / settings.nutrition_reference_seed_path


def _load_seed_payload() -> dict[str, Any]:
    path = _seed_path()
    if not path.exists():
        raise FileNotFoundError(f"Nutrition seed file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(value)
    except Exception:
        return None


def initialize_nutrition_reference(force: bool = False) -> None:
    settings = get_settings()
    _NUTRITION_STATE["enabled"] = _nutrition_enabled()

    if not _nutrition_enabled():
        _NUTRITION_STATE["initialized"] = False
        _NUTRITION_STATE["last_error"] = None
        return

    if _NUTRITION_STATE["initialized"] and not force:
        return

    psycopg = _load_psycopg()
    if psycopg is None:
        _NUTRITION_STATE["initialized"] = False
        _NUTRITION_STATE["last_error"] = "psycopg is not installed"
        return

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS nutrition_reference_versions (
                        version_id BIGSERIAL PRIMARY KEY,
                        source_name TEXT NOT NULL,
                        source_version TEXT NOT NULL,
                        notes TEXT,
                        loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        record_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS nutrition_reference_items (
                        id BIGSERIAL PRIMARY KEY,
                        ingredient_key TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        source_version TEXT NOT NULL,
                        version_id BIGINT NOT NULL REFERENCES nutrition_reference_versions(version_id) ON DELETE CASCADE,
                        calories_per_100g DOUBLE PRECISION,
                        protein_g_per_100g DOUBLE PRECISION,
                        fat_g_per_100g DOUBLE PRECISION,
                        carbs_g_per_100g DOUBLE PRECISION,
                        assumed_serving_g DOUBLE PRECISION,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_nutrition_reference_items_unique
                    ON nutrition_reference_items (ingredient_key, source_name, source_version)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_nutrition_reference_items_ingredient_key
                    ON nutrition_reference_items (ingredient_key)
                    """
                )
        refresh_nutrition_reference(force=force)
        _NUTRITION_STATE["initialized"] = True
        _NUTRITION_STATE["last_error"] = None
    except Exception as exc:
        _NUTRITION_STATE["initialized"] = False
        _NUTRITION_STATE["latest_version"] = None
        _NUTRITION_STATE["last_error"] = str(exc)


def refresh_nutrition_reference(force: bool = False) -> dict[str, Any]:
    settings = get_settings()
    psycopg = _load_psycopg()
    if not settings.database_url or psycopg is None:
        raise RuntimeError("Nutrition reference refresh requires a configured PostgreSQL connection.")

    payload = _load_seed_payload()
    source_name = str(payload.get("source_name") or "curated_prototype_reference")
    source_version = str(payload.get("source_version") or "unknown")
    notes = str(payload.get("notes") or "").strip() or None
    items = payload.get("items") or []
    if not isinstance(items, list) or not items:
        raise RuntimeError("Nutrition seed payload does not contain items.")

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, record_count
                FROM nutrition_reference_versions
                WHERE source_name = %s AND source_version = %s
                ORDER BY loaded_at DESC
                LIMIT 1
                """,
                (source_name, source_version),
            )
            existing = cur.fetchone()
            if existing and not force:
                version_id, record_count = existing
                result = {
                    "source_name": source_name,
                    "source_version": source_version,
                    "version_id": int(version_id),
                    "record_count": int(record_count),
                    "skipped": True,
                }
                _NUTRITION_STATE["latest_version"] = result
                _NUTRITION_STATE["initialized"] = True
                _NUTRITION_STATE["last_error"] = None
                return result

            if existing and force:
                cur.execute(
                    """
                    DELETE FROM nutrition_reference_versions
                    WHERE source_name = %s AND source_version = %s
                    """,
                    (source_name, source_version),
                )

            cur.execute(
                """
                INSERT INTO nutrition_reference_versions (
                    source_name,
                    source_version,
                    notes,
                    record_count
                )
                VALUES (%s, %s, %s, %s)
                RETURNING version_id
                """,
                (source_name, source_version, notes, len(items)),
            )
            version_id = int(cur.fetchone()[0])

            inserted = 0
            for item in items:
                ingredient_key = str(item.get("ingredient_key") or "").strip().lower()
                display_name = str(item.get("display_name") or ingredient_key).strip()
                if not ingredient_key or not display_name:
                    continue
                cur.execute(
                    """
                    INSERT INTO nutrition_reference_items (
                        ingredient_key,
                        display_name,
                        source_name,
                        source_version,
                        version_id,
                        calories_per_100g,
                        protein_g_per_100g,
                        fat_g_per_100g,
                        carbs_g_per_100g,
                        assumed_serving_g,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        ingredient_key,
                        display_name,
                        source_name,
                        source_version,
                        version_id,
                        _safe_float(item.get("calories_per_100g")),
                        _safe_float(item.get("protein_g_per_100g")),
                        _safe_float(item.get("fat_g_per_100g")),
                        _safe_float(item.get("carbs_g_per_100g")),
                        _safe_float(item.get("assumed_serving_g")),
                        json.dumps({"seed_path": str(_seed_path())}),
                    ),
                )
                inserted += 1

    result = {
        "source_name": source_name,
        "source_version": source_version,
        "version_id": version_id,
        "record_count": inserted,
        "skipped": False,
    }
    _NUTRITION_STATE["latest_version"] = result
    _NUTRITION_STATE["initialized"] = True
    _NUTRITION_STATE["last_error"] = None
    return result


def lookup_nutrition_references(ingredients: list[str]) -> list[dict[str, Any]]:
    initialize_nutrition_reference()
    settings = get_settings()
    psycopg = _load_psycopg()
    if not settings.database_url or psycopg is None or not ingredients:
        return []

    normalized = sorted({str(item).strip().lower() for item in ingredients if str(item).strip()})
    if not normalized:
        return []

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (ingredient_key)
                        ingredient_key,
                        display_name,
                        source_name,
                        source_version,
                        calories_per_100g,
                        protein_g_per_100g,
                        fat_g_per_100g,
                        carbs_g_per_100g,
                        assumed_serving_g
                    FROM nutrition_reference_items
                    WHERE ingredient_key = ANY(%s)
                    ORDER BY ingredient_key, version_id DESC
                    """,
                    (normalized,),
                )
                rows = cur.fetchall()
    except Exception as exc:
        _NUTRITION_STATE["last_error"] = str(exc)
        return []

    result: list[dict[str, Any]] = []
    for row in rows:
        (
            ingredient_key,
            display_name,
            source_name,
            source_version,
            calories_per_100g,
            protein_g_per_100g,
            fat_g_per_100g,
            carbs_g_per_100g,
            assumed_serving_g,
        ) = row
        result.append(
            {
                "ingredient_key": str(ingredient_key),
                "display_name": str(display_name),
                "source_name": str(source_name),
                "source_version": str(source_version),
                "calories_per_100g": _safe_float(calories_per_100g),
                "protein_g_per_100g": _safe_float(protein_g_per_100g),
                "fat_g_per_100g": _safe_float(fat_g_per_100g),
                "carbs_g_per_100g": _safe_float(carbs_g_per_100g),
                "assumed_serving_g": _safe_float(assumed_serving_g),
            }
        )
    return result


def estimate_from_references(
    ingredients: list[str],
    ingredient_weights: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    refs = lookup_nutrition_references(ingredients)
    if not refs:
        return None

    mid = 0.0
    matched: list[str] = []
    matched_weights: list[float] = []
    for ref in refs:
        calories = _safe_float(ref.get("calories_per_100g"))
        serving = _safe_float(ref.get("assumed_serving_g"))
        if calories is None or serving is None:
            continue
        ingredient_key = str(ref.get("ingredient_key"))
        weight = 1.0
        if ingredient_weights is not None:
            weight = float(ingredient_weights.get(ingredient_key, 0.65))
            weight = max(0.35, min(1.0, weight))
        mid += calories * serving / 100.0 * weight
        matched.append(ingredient_key)
        matched_weights.append(weight)

    if mid <= 0:
        return None

    result = {
        "matched_ingredients": matched,
        "matched_weight_sum": round(sum(matched_weights), 3),
        "avg_match_weight": round(sum(matched_weights) / len(matched_weights), 3) if matched_weights else 0.0,
        "calories_low": round(max(40.0, mid * 0.8), 1),
        "calories_mid": round(mid, 1),
        "calories_high": round(max(mid, mid * 1.25), 1),
        "confidence_bonus": min(0.2, 0.05 * len(matched)),
        "source_versions": sorted(
            {
                f"{ref['source_name']}:{ref['source_version']}"
                for ref in refs
                if ref.get("source_name") and ref.get("source_version")
            }
        ),
    }
    return result


def load_nutrition_reference_stats() -> dict[str, Any]:
    initialize_nutrition_reference()
    settings = get_settings()
    psycopg = _load_psycopg()
    result = {
        "enabled": _nutrition_enabled(),
        "initialized": bool(_NUTRITION_STATE["initialized"]),
        "database_url_present": bool(settings.database_url),
        "seed_path": str(_seed_path()),
        "latest_version": _NUTRITION_STATE["latest_version"],
        "row_counts": {},
        "source_counts": {},
        "last_error": _NUTRITION_STATE["last_error"],
    }
    if not settings.database_url or psycopg is None:
        return result

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                for table_name in ["nutrition_reference_versions", "nutrition_reference_items"]:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    result["row_counts"][table_name] = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT source_name, COUNT(*)
                    FROM nutrition_reference_items
                    GROUP BY source_name
                    ORDER BY source_name
                    """
                )
                result["source_counts"] = {str(source_name): int(count) for source_name, count in cur.fetchall()}
                cur.execute(
                    """
                    SELECT source_name, source_version, version_id, loaded_at, record_count
                    FROM nutrition_reference_versions
                    ORDER BY loaded_at DESC
                    LIMIT 1
                    """
                )
                latest = cur.fetchone()
                if latest:
                    source_name, source_version, version_id, loaded_at, record_count = latest
                    result["latest_version"] = {
                        "source_name": str(source_name),
                        "source_version": str(source_version),
                        "version_id": int(version_id),
                        "loaded_at": str(loaded_at),
                        "record_count": int(record_count),
                    }
        _NUTRITION_STATE["last_error"] = None
    except Exception as exc:
        result["last_error"] = str(exc)
        _NUTRITION_STATE["last_error"] = str(exc)
    return result

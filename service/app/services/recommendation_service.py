from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from functools import lru_cache
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import CombinationRow, ParsedItem, RecommendRequest, RecommendationRow


@dataclass
class EngineResult:
    engine_used: str
    scores: np.ndarray


RERANK_FEATURE_NAMES = [
    "semantic_score",
    "tfidf_score",
    "rule_score",
    "token_jaccard",
    "query_overlap",
    "item_overlap",
    "section_match",
    "ingredient_overlap",
    "has_description",
    "price_present",
    "name_overlap",
]
RECOMMENDATION_CACHE_SCHEMA_VERSION = "recommendation_runtime_v2"


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _item_text(item: ParsedItem) -> str:
    parts = [item.dish_name or "", item.description or "", item.section or ""]
    return " ".join(part.strip() for part in parts if part).strip()


def _query_text(request: RecommendRequest) -> str:
    parts = []
    if request.craving_text:
        parts.append(request.craving_text)
    if request.liked_terms:
        parts.append(" ".join(request.liked_terms))
    if request.preferred_sections:
        parts.append(" ".join(request.preferred_sections))
    return " ".join(parts).strip()


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _tokenize(text: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(_normalize_text(text))]


def _safe_section(text: str | None) -> str:
    return _normalize_text(text).lower()


def _calorie_limit_status(request: RecommendRequest, item: ParsedItem) -> str:
    if request.max_calories is None or item.calories_mid is None:
        return "unknown"
    if float(item.calories_mid) <= float(request.max_calories):
        return "within"

    confidence = float(item.nutrition_confidence or 0.0)
    if confidence >= 0.75:
        return "exceeds"
    if item.calories_low is not None and float(item.calories_low) > float(request.max_calories):
        return "exceeds"
    return "uncertain"


def _try_sentence_transformer(texts: list[str], query_text: str) -> EngineResult | None:
    backend_kind, model = _load_sentence_transformer_model()
    if backend_kind == "sentence_transformers":
        matrix = model.encode(texts, normalize_embeddings=True)
        query_vec = model.encode([query_text], normalize_embeddings=True)
    else:
        matrix = np.asarray(list(model.embed(texts)), dtype=float)
        query_vec = np.asarray(list(model.embed([query_text])), dtype=float)
        matrix = _normalize_embedding_matrix(matrix)
        query_vec = _normalize_embedding_matrix(query_vec)
    scores = cosine_similarity(query_vec, matrix)[0]
    return EngineResult(engine_used="sentence_transformer", scores=np.asarray(scores))


@lru_cache(maxsize=1)
def _load_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
        return "sentence_transformers", SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        try:
            from fastembed import TextEmbedding
        except Exception as exc:
            raise RuntimeError(
                "No semantic embedding backend is installed. Install sentence-transformers or fastembed for the sentence_transformer engine."
            ) from exc
        try:
            return "fastembed", TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        except Exception as exc:
            raise RuntimeError(
                "Could not load the semantic embedding model. Check internet access for the first download or make sure the model is cached locally."
            ) from exc


def _normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _tfidf_scores(texts: list[str], query_text: str) -> EngineResult:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts + [query_text])
    item_matrix = matrix[:-1]
    query_vec = matrix[-1]
    scores = cosine_similarity(query_vec, item_matrix)[0]
    return EngineResult(engine_used="tfidf", scores=np.asarray(scores))


def _semantic_scores(texts: list[str], query_text: str, engine: str) -> EngineResult:
    if not query_text:
        return EngineResult(engine_used="none", scores=np.zeros(len(texts), dtype=float))

    if engine in {"auto", "sentence_transformer", "catboost_reranker"}:
        try:
            st_result = _try_sentence_transformer(texts, query_text)
        except RuntimeError:
            st_result = None
            if engine in {"sentence_transformer", "catboost_reranker"}:
                raise
        if st_result is not None:
            return st_result

    return _tfidf_scores(texts, query_text)


def _apply_filters(request: RecommendRequest, items: Iterable[ParsedItem]) -> list[ParsedItem]:
    excluded_allergens = {x.lower() for x in request.excluded_allergens}
    disliked_terms = {x.lower() for x in request.disliked_terms}
    excluded_sections = {x.lower() for x in request.excluded_sections}
    required_diet_flags = {x.lower() for x in request.required_diet_flags}

    kept: list[ParsedItem] = []
    for item in items:
        item_text = _item_text(item).lower()
        item_section = (item.section or "").lower()
        item_allergens = {x.lower() for x in item.explicit_allergens}

        if excluded_allergens & item_allergens:
            continue
        if excluded_sections and item_section in excluded_sections:
            continue
        if any(term in item_text for term in disliked_terms):
            continue
        if request.max_price is not None and item.price_value is not None and item.price_value > request.max_price:
            continue
        if _calorie_limit_status(request, item) == "exceeds":
            continue
        if required_diet_flags:
            item_flags = {k.lower() for k, v in item.diet_flags.items() if v}
            if not required_diet_flags.issubset(item_flags):
                continue
        kept.append(item)
    return kept


def _rule_score(request: RecommendRequest, item: ParsedItem) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if request.preferred_sections and item.section:
        if item.section.lower() in {x.lower() for x in request.preferred_sections}:
            score += 0.15
            reasons.append("preferred section")

    if item.description:
        score += 0.05
        reasons.append("has description")

    if item.parser_confidence is not None:
        score += 0.15 * float(item.parser_confidence)
        reasons.append("parser confidence")

    if request.max_price is not None and item.price_value is not None:
        if item.price_value <= request.max_price:
            score += 0.10
            reasons.append("within budget")

    if request.max_calories is not None and item.calories_mid is not None:
        calorie_status = _calorie_limit_status(request, item)
        if calorie_status == "within":
            score += 0.14
            reasons.append("within calorie target")
        elif calorie_status == "uncertain":
            score += 0.05
            reasons.append("possible calorie fit")

    if item.nutrition_confidence is not None:
        score += 0.10 * float(item.nutrition_confidence)
        reasons.append("nutrition estimate available")

    if request.required_diet_flags:
        matched_flags = [flag for flag in request.required_diet_flags if item.diet_flags.get(flag)]
        if matched_flags:
            score += 0.18
            reasons.append("diet filter match")

    if request.liked_terms:
        item_text = _item_text(item).lower()
        matches = [term for term in request.liked_terms if term.lower() in item_text]
        if matches:
            score += 0.20
            reasons.append("liked terms match")

    return score, reasons


def _match_label(score: float) -> str:
    if score >= 0.68:
        return "Strong match"
    if score >= 0.40:
        return "Good match"
    return "Weak match"


def _enabled_flags(item: ParsedItem) -> list[str]:
    return [key for key, value in item.diet_flags.items() if value]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_catboost_reranker_path() -> Path:
    rel_path = os.getenv(
        "PICKMEAL_RECOMMENDATION_RERANKER_PATH",
        "reports/recommendation_catboost_reranker_v1/catboost_reranker.cbm",
    )
    return _project_root() / rel_path


def recommendation_cache_context() -> dict[str, str]:
    return {
        "cache_schema_version": RECOMMENDATION_CACHE_SCHEMA_VERSION,
        "semantic_backend_chain": "sentence_transformers/all-MiniLM-L6-v2|fastembed/BAAI-bge-small-en-v1.5|tfidf",
        "catboost_reranker_path": str(_default_catboost_reranker_path()),
    }


@lru_cache(maxsize=1)
def _load_catboost_reranker():
    path = _default_catboost_reranker_path()
    if not path.exists():
        raise RuntimeError(
            f"CatBoost reranker model is missing at {path}. Train it first or switch the recommendation engine."
        )
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise RuntimeError(
            "catboost is not installed in the current environment."
        ) from exc
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _catboost_rerank_features(
    request: RecommendRequest,
    item: ParsedItem,
    *,
    semantic_score: float,
    rule_score: float,
    query_text: str,
) -> list[float]:
    query_tokens = set(_tokenize(query_text))
    item_text = _item_text(item)
    item_tokens = set(_tokenize(item_text))
    overlap = len(query_tokens & item_tokens)
    union = len(query_tokens | item_tokens) or 1
    section_match = int(
        bool(request.preferred_sections)
        and _safe_section(item.section) in {_safe_section(section) for section in request.preferred_sections}
    )
    query_ingredient_tokens = set(_tokenize(" ".join(request.liked_terms)))
    item_ingredient_tokens = set(_tokenize(" ".join(item.ingredient_hints)))
    dish_name_tokens = set(_tokenize(item.dish_name))
    tfidf_score = float(_tfidf_scores([item_text], query_text).scores[0]) if query_text else 0.0
    return [
        float(semantic_score),
        tfidf_score,
        float(rule_score),
        overlap / union,
        overlap / (len(query_tokens) or 1),
        overlap / (len(item_tokens) or 1),
        float(section_match),
        float(len(query_ingredient_tokens & item_ingredient_tokens)),
        float(int(bool(_normalize_text(item.description)))),
        float(int(item.price_value is not None)),
        float(len(dish_name_tokens & query_tokens)),
    ]


def _apply_catboost_rerank(
    request: RecommendRequest,
    rows: list[RecommendationRow],
    items_by_id: dict[str, ParsedItem],
    query_text: str,
) -> tuple[str, list[RecommendationRow]]:
    model = _load_catboost_reranker()
    feature_rows: list[list[float]] = []
    ordered_rows: list[RecommendationRow] = []
    for row in rows:
        item = items_by_id.get(row.local_id)
        if item is None:
            continue
        feature_rows.append(
            _catboost_rerank_features(
                request,
                item,
                semantic_score=float(row.semantic_score),
                rule_score=float(row.rule_score),
                query_text=query_text,
            )
        )
        ordered_rows.append(row)

    if not feature_rows:
        return "catboost_reranker", rows

    feature_matrix = np.asarray(feature_rows, dtype=float)
    raw_scores = np.asarray(model.predict(feature_matrix, prediction_type="RawFormulaVal"), dtype=float).reshape(-1)
    if raw_scores.size == 0:
        return "catboost_reranker", rows
    if float(raw_scores.max()) == float(raw_scores.min()):
        rerank_scores = np.full_like(raw_scores, 0.5, dtype=float)
    else:
        rerank_scores = (raw_scores - float(raw_scores.min())) / (float(raw_scores.max()) - float(raw_scores.min()))
    reranked: list[RecommendationRow] = []
    for row, score in zip(ordered_rows, rerank_scores):
        blended_score = 0.65 * float(row.score) + 0.35 * float(score)
        reranked.append(
            row.model_copy(
                update={
                    "score": round(blended_score, 4),
                    "match_label": _match_label(blended_score),
                    "reasons": row.reasons + ["catboost reranker"],
                }
            )
        )
    reranked.sort(key=lambda rec: rec.score, reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row.rank = rank
    return "catboost_reranker", reranked


def _combo_score(
    request: RecommendRequest,
    combo_items: list[ParsedItem],
    item_scores: dict[str, float],
) -> tuple[float, list[str], float | None, float | None]:
    total_price = None
    if all(item.price_value is not None for item in combo_items):
        total_price = round(sum(float(item.price_value) for item in combo_items), 2)

    total_calories = None
    if all(item.calories_mid is not None for item in combo_items):
        total_calories = round(sum(float(item.calories_mid) for item in combo_items), 1)

    reasons: list[str] = []
    score = float(np.mean([item_scores[item.local_id] for item in combo_items]))

    sections = [item.section for item in combo_items if item.section]
    if len(set(sections)) >= 2:
        score += 0.06
        reasons.append("section diversity")

    if request.combo_budget is not None and total_price is not None and total_price <= request.combo_budget:
        score += 0.12
        reasons.append("within combo budget")
        if request.combo_budget > 0:
            budget_ratio = total_price / request.combo_budget
            if 0.65 <= budget_ratio <= 1.0:
                score += 0.05
                reasons.append("good budget usage")

    if request.combo_max_calories is not None and total_calories is not None and total_calories <= request.combo_max_calories:
        score += 0.10
        reasons.append("within combo calorie target")

    if all(item.nutrition_confidence is not None for item in combo_items):
        avg_conf = float(np.mean([float(item.nutrition_confidence) for item in combo_items]))
        score += 0.05 * avg_conf
        reasons.append("nutrition estimates available")

    return score, reasons, total_price, total_calories


def build_combo_rows(request: RecommendRequest, candidates: list[ParsedItem], item_rows: list[RecommendationRow]) -> list[CombinationRow]:
    if request.combo_budget is None and request.combo_max_calories is None:
        return []

    min_items = max(2, int(request.combo_min_items))
    max_items = max(min_items, int(request.combo_max_items))
    ranked_items = sorted(candidates, key=lambda item: next((row.score for row in item_rows if row.local_id == item.local_id), 0.0), reverse=True)
    pool = ranked_items[: min(len(ranked_items), 8)]
    if len(pool) < min_items:
        return []

    item_scores = {row.local_id: row.score for row in item_rows}
    combos: list[CombinationRow] = []
    seen: set[tuple[str, ...]] = set()

    for size in range(min_items, min(max_items, len(pool)) + 1):
        for combo in combinations(pool, size):
            key = tuple(sorted(item.local_id for item in combo))
            if key in seen:
                continue
            seen.add(key)

            if request.combo_budget is not None:
                if any(item.price_value is None for item in combo):
                    continue
                total_price = sum(float(item.price_value) for item in combo)
                if total_price > request.combo_budget:
                    continue

            if request.combo_max_calories is not None:
                if any(item.calories_mid is None for item in combo):
                    continue
                total_calories = sum(float(item.calories_mid) for item in combo)
                if total_calories > request.combo_max_calories:
                    continue

            score, reasons, total_price, total_calories = _combo_score(request, list(combo), item_scores)
            combos.append(
                CombinationRow(
                    rank=0,
                    item_ids=[item.local_id for item in combo],
                    dish_names=[item.dish_name for item in combo],
                    sections=[item.section for item in combo if item.section],
                    total_price=total_price,
                    total_calories=total_calories,
                    score=round(score, 4),
                    match_label=_match_label(score),
                    reasons=reasons,
                )
            )

    combos.sort(key=lambda row: row.score, reverse=True)
    combos = combos[: max(1, request.top_k)]
    for idx, row in enumerate(combos, start=1):
        row.rank = idx
    return combos


def recommend_items(request: RecommendRequest) -> tuple[str, int, list[RecommendationRow], list[CombinationRow]]:
    candidates = _apply_filters(request, request.items)
    if not candidates:
        return "none", 0, [], []

    texts = [_item_text(item) for item in candidates]
    query_text = _query_text(request)
    engine_result = _semantic_scores(texts, query_text, request.engine)

    all_rows: list[RecommendationRow] = []
    for idx, item in enumerate(candidates):
        rule_score, reasons = _rule_score(request, item)
        semantic_score = float(engine_result.scores[idx])
        total_score = 0.60 * semantic_score + 0.40 * rule_score
        all_rows.append(
            RecommendationRow(
                rank=0,
                local_id=item.local_id,
                dish_name=item.dish_name,
                section=item.section,
                price_value=item.price_value,
                calories_mid=item.calories_mid,
                diet_flags=_enabled_flags(item),
                match_label=_match_label(total_score),
                score=round(total_score, 4),
                semantic_score=round(semantic_score, 4),
                rule_score=round(rule_score, 4),
                reasons=reasons,
            )
        )

    all_rows.sort(key=lambda row: row.score, reverse=True)
    top_k = max(1, request.top_k)
    rows = all_rows[:top_k]

    for rank, row in enumerate(rows, start=1):
        row.rank = rank

    engine_used = engine_result.engine_used
    if request.engine == "catboost_reranker" and query_text:
        items_by_id = {item.local_id: item for item in candidates}
        engine_used, reranked_rows = _apply_catboost_rerank(request, all_rows, items_by_id, query_text)
        all_rows = reranked_rows
        rows = all_rows[:top_k]

    combo_rows = build_combo_rows(request, candidates, all_rows)

    return engine_used, len(candidates), rows, combo_rows

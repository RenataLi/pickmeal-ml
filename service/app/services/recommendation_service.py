from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import ParsedItem, RecommendRequest, RecommendationRow


@dataclass
class EngineResult:
    engine_used: str
    scores: np.ndarray


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


def _try_sentence_transformer(texts: list[str], query_text: str) -> EngineResult | None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    # Small multilingual model is a good later option, but we avoid hard-coding downloads here.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    matrix = model.encode(texts, normalize_embeddings=True)
    query_vec = model.encode([query_text], normalize_embeddings=True)
    scores = cosine_similarity(query_vec, matrix)[0]
    return EngineResult(engine_used="sentence_transformer", scores=np.asarray(scores))


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

    if engine in {"auto", "sentence_transformer"}:
        st_result = _try_sentence_transformer(texts, query_text)
        if st_result is not None:
            return st_result
        if engine == "sentence_transformer":
            raise RuntimeError("sentence-transformers is not installed.")

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
        if request.max_calories is not None and item.calories_mid is not None and item.calories_mid > request.max_calories:
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
        if item.calories_mid <= request.max_calories:
            score += 0.14
            reasons.append("within calorie target")

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


def recommend_items(request: RecommendRequest) -> tuple[str, int, list[RecommendationRow]]:
    candidates = _apply_filters(request, request.items)
    if not candidates:
        return "none", 0, []

    texts = [_item_text(item) for item in candidates]
    query_text = _query_text(request)
    engine_result = _semantic_scores(texts, query_text, request.engine)

    rows: list[RecommendationRow] = []
    for idx, item in enumerate(candidates):
        rule_score, reasons = _rule_score(request, item)
        semantic_score = float(engine_result.scores[idx])
        total_score = 0.60 * semantic_score + 0.40 * rule_score
        rows.append(
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

    rows.sort(key=lambda row: row.score, reverse=True)
    top_k = max(1, request.top_k)
    rows = rows[:top_k]

    for rank, row in enumerate(rows, start=1):
        row.rank = rank

    return engine_result.engine_used, len(candidates), rows

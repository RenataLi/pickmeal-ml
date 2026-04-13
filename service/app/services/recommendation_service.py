from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schemas import CombinationRow, ParsedItem, RecommendRequest, RecommendationRow


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

    combo_rows = build_combo_rows(request, candidates, rows)

    return engine_result.engine_used, len(candidates), rows, combo_rows

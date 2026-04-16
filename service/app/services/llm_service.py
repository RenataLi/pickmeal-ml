from __future__ import annotations

import json
import re
from typing import Any

import requests

from ..config import get_settings
from ..schemas import ParsedItem, RAGEvidenceRow
from .rag_service import retrieve_rag_for_items


_LLM_STATE: dict[str, str | None] = {
    "last_error": None,
}


def llm_is_configured() -> bool:
    settings = get_settings()
    return bool(settings.llm_enabled and settings.llm_base_url and settings.llm_api_key and settings.llm_model)


def load_llm_stats() -> dict[str, Any]:
    settings = get_settings()
    return {
        "enabled": bool(settings.llm_enabled),
        "configured": llm_is_configured(),
        "base_url_present": bool(settings.llm_base_url),
        "api_key_present": bool(settings.llm_api_key),
        "model": settings.llm_model,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_items_per_request": settings.llm_max_items_per_request,
        "last_error": _LLM_STATE["last_error"],
    }


def _build_prompt_items(items: list[ParsedItem], evidence_by_item: dict[str, list[RAGEvidenceRow]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
        evidence_rows = evidence_by_item.get(item.local_id, [])
        payload.append(
            {
                "local_id": item.local_id,
                "dish_name": item.dish_name,
                "section": item.section,
                "description": item.description,
                "ingredient_hints": item.ingredient_hints,
                "explicit_allergens": item.explicit_allergens,
                "diet_flags": item.diet_flags,
                "calories_low": item.calories_low,
                "calories_mid": item.calories_mid,
                "calories_high": item.calories_high,
                "nutrition_confidence": item.nutrition_confidence,
                "parser_confidence": item.parser_confidence,
                "retrieved_evidence": [
                    {
                        "source_type": row.source_type,
                        "source_id": row.source_id,
                        "title": row.title,
                        "similarity": row.similarity,
                        "content_preview": row.content_preview,
                    }
                    for row in evidence_rows
                ],
            }
        )
    return payload


def _strip_json_wrapper(text: str) -> str:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return candidate


def _parse_response_items(content: str) -> list[dict[str, Any]]:
    candidate = _strip_json_wrapper(content)
    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse LLM JSON output: {exc}") from exc

    if isinstance(parsed, dict):
        rows = parsed.get("items", [])
    elif isinstance(parsed, list):
        rows = parsed
    else:
        rows = []
    if not isinstance(rows, list):
        raise RuntimeError("LLM output does not contain an items list")
    return rows


def _request_chat_completion(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    if not llm_is_configured():
        raise RuntimeError("LLM is not configured. Set PICKMEAL_LLM_ENABLED, PICKMEAL_LLM_BASE_URL, PICKMEAL_LLM_API_KEY, and PICKMEAL_LLM_MODEL.")

    base_url = settings.llm_base_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "messages": messages,
    }
    response = requests.post(endpoint, headers=headers, json=payload, timeout=settings.llm_timeout_seconds)
    if not response.ok:
        raise RuntimeError(f"LLM request failed: {response.status_code} {response.text[:400]}")
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected LLM response format: {exc}") from exc
    return content


def _extract_field(row: dict[str, Any], names: list[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalized_rows(rows: list[dict[str, Any]], items: list[ParsedItem]) -> dict[str, dict[str, Any]]:
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        local_id = _extract_field(row, ["local_id", "item_local_id", "id"])
        if local_id:
            row_by_id[local_id] = row

    if len(row_by_id) == len(items):
        return row_by_id

    if len(rows) == len(items):
        for item, row in zip(items, rows):
            if isinstance(row, dict):
                row_by_id.setdefault(item.local_id, row)
    return row_by_id


def _fallback_summary(item: ParsedItem, evidence_rows: list[RAGEvidenceRow]) -> str:
    section = item.section.lower() if item.section else "menu"
    calories = f"{int(item.calories_mid)} kcal estimate" if item.calories_mid is not None else "broad calorie estimate"
    ingredient_text = ", ".join(item.ingredient_hints[:3]) if item.ingredient_hints else ""
    if ingredient_text:
        return f"{item.dish_name} from the {section} section with likely ingredients such as {ingredient_text}; {calories}."
    if evidence_rows:
        return f"{item.dish_name} from the {section} section, grounded with similar annotated menu examples; {calories}."
    return f"{item.dish_name} from the {section} section with {calories}."


def _user_goal_snippets(user_context: str | None) -> list[str]:
    text = (user_context or "").strip().lower()
    if not text:
        return []

    snippets: list[str] = []
    if any(token in text for token in ["vegetarian", "vegan", "plant-based", "plant based"]):
        snippets.append("plant-based preferences")
    if any(token in text for token in ["gluten", "gluten-free", "gf"]):
        snippets.append("gluten-aware filtering")
    if any(token in text for token in ["dairy", "lactose", "milk-free", "milk free"]):
        snippets.append("dairy-aware filtering")
    if any(token in text for token in ["allergy", "allergen", "safe", "avoid"]):
        snippets.append("allergen-aware filtering")
    if any(token in text for token in ["light", "lighter", "low calorie", "not too heavy", "healthy"]):
        snippets.append("lighter meal preferences")
    if any(token in text for token in ["protein", "high-protein", "high protein", "gym", "post-workout"]):
        snippets.append("protein-focused choices")
    return snippets


def _clean_generated_text(text: str | None, user_context: str | None) -> str | None:
    if text is None:
        return None
    cleaned = " ".join(str(text).split())
    context = (user_context or "").strip()
    if context:
        patterns = [
            rf"(?i)\band the user context:\s*{re.escape(context)}\.?",
            rf"(?i)\buser context:\s*{re.escape(context)}\.?",
            rf"(?i)\bthe user context:\s*{re.escape(context)}\.?",
            re.escape(context),
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = cleaned.strip(" -:;,.")
    return cleaned or None


def _fallback_why(item: ParsedItem, user_context: str | None) -> str:
    positive_flags = [flag.replace("_", " ") for flag, enabled in item.diet_flags.items() if enabled]
    goal_snippets = _user_goal_snippets(user_context)
    notes: list[str] = []

    if positive_flags:
        notes.append(f"Parsed menu signals suggest {', '.join(positive_flags[:3])}.")
    if item.calories_mid is not None and item.calories_mid <= 450:
        notes.append("Its calorie estimate sits on the lighter side.")
    elif item.calories_mid is not None and item.calories_mid <= 650:
        notes.append("Its calorie estimate stays in a moderate range.")
    if goal_snippets:
        notes.append(f"It aligns with {', '.join(goal_snippets[:2])}.")
    elif item.ingredient_hints:
        notes.append(f"It highlights likely ingredients such as {', '.join(item.ingredient_hints[:2])}.")

    if notes:
        return " ".join(notes[:2])
    return "It is included because the parsed menu text produced a usable structured dish entry."


def _fallback_caution(item: ParsedItem) -> str:
    cautions: list[str] = []
    if item.explicit_allergens:
        cautions.append(f"Possible allergens: {', '.join(item.explicit_allergens[:4])}.")
    if item.nutrition_confidence is not None and item.nutrition_confidence < 0.75:
        cautions.append("Nutrition estimate is heuristic and may be broad.")
    if not item.description:
        cautions.append("Description is missing, so some details may be uncertain.")
    if not cautions:
        cautions.append("Use the menu text or staff confirmation for exact ingredients and allergen handling.")
    return " ".join(cautions)


def enrich_items_with_llm(
    items: list[ParsedItem],
    user_context: str | None = None,
    top_k: int = 5,
) -> tuple[str, str, list[ParsedItem], list[RAGEvidenceRow]]:
    settings = get_settings()
    if not items:
        return "llm", settings.llm_model or "unconfigured", [], []
    if not llm_is_configured():
        raise RuntimeError("LLM enrichment is disabled or not configured.")

    limited_items = items[: max(1, min(top_k, settings.llm_max_items_per_request))]
    evidence_by_item, retrieval_rows = retrieve_rag_for_items(
        limited_items,
        user_context=user_context,
        top_k=settings.rag_top_k,
    )
    prompt_items = _build_prompt_items(limited_items, evidence_by_item)
    context_text = user_context.strip() if user_context else ""

    system_prompt = (
        "You are helping a restaurant menu recommendation app. "
        "You will receive structured dish items plus retrieved evidence from the app knowledge base. "
        "Return only valid JSON. "
        "For every item, always return non-empty strings for llm_summary, llm_why_it_fits, and llm_caution_note, "
        "and always echo the exact local_id from input. "
        "Use the structured item fields first, then use retrieved evidence only as supporting context. "
        "Treat user_context as a soft preference note, not as text to repeat back. "
        "Do not quote or copy user_context verbatim into llm_why_it_fits. "
        "Do not mention the phrases user context, preference note, or the user's preference. "
        "Instead, explain fit directly through dish properties such as section, ingredients, diet flags, allergens, or calorie range. "
        "Do not invent ingredients, allergens, diets, or calorie facts beyond the provided fields and retrieved evidence. "
        "If uncertainty exists, say so briefly in llm_caution_note instead of leaving fields blank."
    )
    user_prompt = {
        "user_context": context_text or None,
        "items": prompt_items,
        "output_format": {
            "items": [
                {
                    "local_id": "same as input",
                    "llm_summary": "short grounded summary",
                    "llm_why_it_fits": "short grounded fit explanation",
                    "llm_caution_note": "short grounded caution note",
                }
            ]
        },
    }

    try:
        content = _request_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ]
        )
        rows = _parse_response_items(content)
        row_by_id = _normalized_rows(rows, limited_items)
        enriched_items: list[ParsedItem] = []
        for item in limited_items:
            row = row_by_id.get(item.local_id, {})
            evidence_rows = evidence_by_item.get(item.local_id, [])
            summary = _clean_generated_text(_extract_field(row, ["llm_summary", "summary", "short_summary"]), context_text)
            why_it_fits = _clean_generated_text(_extract_field(row, ["llm_why_it_fits", "why_it_fits", "fit_reason"]), context_text)
            caution_note = _clean_generated_text(_extract_field(row, ["llm_caution_note", "caution_note", "caution"]), context_text)
            enriched_items.append(
                item.model_copy(
                    update={
                        "llm_summary": summary or _fallback_summary(item, evidence_rows),
                        "llm_why_it_fits": why_it_fits or _fallback_why(item, context_text),
                        "llm_caution_note": caution_note or _fallback_caution(item),
                    }
                )
            )
        _LLM_STATE["last_error"] = None
        return "rag_openai_compatible", settings.llm_model or "unknown", enriched_items, retrieval_rows
    except Exception as exc:
        _LLM_STATE["last_error"] = str(exc)
        raise

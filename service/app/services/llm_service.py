from __future__ import annotations

import json
from typing import Any

import requests

from ..config import get_settings
from ..schemas import ParsedItem


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


def _build_prompt_items(items: list[ParsedItem]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
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


def enrich_items_with_llm(items: list[ParsedItem], user_context: str | None = None, top_k: int = 5) -> tuple[str, str, list[ParsedItem]]:
    settings = get_settings()
    if not items:
        return "llm", settings.llm_model or "unconfigured", []
    if not llm_is_configured():
        raise RuntimeError("LLM enrichment is disabled or not configured.")

    limited_items = items[: max(1, min(top_k, settings.llm_max_items_per_request))]
    prompt_items = _build_prompt_items(limited_items)
    context_text = user_context.strip() if user_context else ""

    system_prompt = (
        "You are helping a restaurant menu recommendation app. "
        "You will receive structured dish items. "
        "Return only valid JSON. "
        "For each item, write concise user-facing fields: "
        "llm_summary, llm_why_it_fits, llm_caution_note. "
        "Do not invent ingredients, allergens, diets, or calorie facts beyond the provided fields. "
        "If uncertainty exists, mention it briefly in llm_caution_note."
    )
    user_prompt = {
        "user_context": context_text or None,
        "items": prompt_items,
        "output_format": {
            "items": [
                {
                    "local_id": "same as input",
                    "llm_summary": "short summary",
                    "llm_why_it_fits": "why it may fit the user context",
                    "llm_caution_note": "short caution or uncertainty note",
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
        row_by_id = {str(row.get("local_id")): row for row in rows if row.get("local_id")}
        enriched_items: list[ParsedItem] = []
        for item in limited_items:
            row = row_by_id.get(item.local_id, {})
            enriched_items.append(
                item.model_copy(
                    update={
                        "llm_summary": row.get("llm_summary"),
                        "llm_why_it_fits": row.get("llm_why_it_fits"),
                        "llm_caution_note": row.get("llm_caution_note"),
                    }
                )
            )
        _LLM_STATE["last_error"] = None
        return "openai_compatible", settings.llm_model or "unknown", enriched_items
    except Exception as exc:
        _LLM_STATE["last_error"] = str(exc)
        raise

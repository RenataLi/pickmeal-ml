from __future__ import annotations

from typing import Any

import requests

from ..config import get_settings


def _clean_url(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip().rstrip("/")
    return text or None


def service_urls() -> dict[str, str | None]:
    settings = get_settings()
    return {
        "ocr": _clean_url(settings.ocr_service_url),
        "parser": _clean_url(settings.parser_service_url),
        "recommendation": _clean_url(settings.recommendation_service_url),
        "llm": _clean_url(settings.llm_service_url),
    }


def service_mode_enabled() -> bool:
    urls = service_urls()
    return any(urls.values())


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    response = requests.post(
        f"{base_url}{path}",
        json=payload,
        timeout=timeout or settings.internal_service_timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"{base_url}{path} returned {response.status_code}: {response.text[:400]}")
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"{base_url}{path} returned a non-JSON response") from exc


def post_multipart(
    base_url: str,
    path: str,
    files: dict[str, Any],
    data: dict[str, Any],
    timeout: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    response = requests.post(
        f"{base_url}{path}",
        files=files,
        data=data,
        timeout=timeout or settings.internal_service_timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"{base_url}{path} returned {response.status_code}: {response.text[:400]}")
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"{base_url}{path} returned a non-JSON response") from exc

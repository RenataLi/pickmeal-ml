from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image

from ..config import get_settings
from ..schemas import OCRLine


PADDLE_LANG_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
}


@dataclass
class OCRRunResult:
    backend: str
    lines: list[OCRLine]


def _normalize_langs(langs_key: str | None) -> list[str]:
    if not langs_key:
        return ["en"]
    langs = [lang.strip().lower() for lang in langs_key.split(",") if lang.strip()]
    return langs or ["en"]


def _with_line_order(rows: list[dict]) -> list[OCRLine]:
    return [
        OCRLine(
            text=row["text"],
            line_order=idx,
            ocr_confidence=row.get("ocr_confidence"),
            bbox_x1=row.get("bbox_x1"),
            bbox_y1=row.get("bbox_y1"),
            bbox_x2=row.get("bbox_x2"),
            bbox_y2=row.get("bbox_y2"),
        )
        for idx, row in enumerate(rows, start=1)
    ]


@lru_cache(maxsize=4)
def _build_reader(langs_key: str):
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "easyocr is not installed. Install it or use /parse/text."
        ) from exc

    langs = [lang.strip() for lang in langs_key.split(",") if lang.strip()]
    if not langs:
        langs = ["en"]
    return easyocr.Reader(langs, gpu=False)


def _configure_paddle_env():
    settings = get_settings()
    hf_cache_dir = os.path.join(settings.paddle_cache_dir, "huggingface")
    modelscope_cache_dir = os.path.join(settings.paddle_cache_dir, "modelscope")
    os.makedirs(settings.paddle_cache_dir, exist_ok=True)
    os.makedirs(settings.paddle_mpl_config_dir, exist_ok=True)
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(modelscope_cache_dir, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", settings.paddle_cache_dir)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("MPLCONFIGDIR", settings.paddle_mpl_config_dir)
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
    os.environ.setdefault("MODELSCOPE_CACHE", modelscope_cache_dir)


def _resolve_paddle_lang(langs_key: str) -> str:
    langs = _normalize_langs(langs_key)
    for lang in langs:
        if lang in PADDLE_LANG_ALIASES:
            return PADDLE_LANG_ALIASES[lang]
    return "en"


@lru_cache(maxsize=4)
def _build_paddleocr_reader(langs_key: str):
    _configure_paddle_env()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "paddleocr is not installed. Install paddleocr and paddlepaddle or switch OCR backend to easyocr."
        ) from exc

    return PaddleOCR(
        lang=_resolve_paddle_lang(langs_key),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def run_easyocr(image_bytes: bytes, langs: str | None = None) -> list[OCRLine]:
    settings = get_settings()
    langs_key = langs or settings.default_ocr_langs
    reader = _build_reader(langs_key)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    result = reader.readtext(image_array, detail=1, paragraph=False)

    rows: list[dict] = []
    for item in result:
        bbox, text, conf = item
        text = str(text).strip()
        if not text:
            continue
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        rows.append(
            {
                "text": text,
                "ocr_confidence": float(conf),
                "bbox_x1": float(min(xs)),
                "bbox_y1": float(min(ys)),
                "bbox_x2": float(max(xs)),
                "bbox_y2": float(max(ys)),
            }
        )
    return _with_line_order(rows)


def run_paddleocr(image_bytes: bytes, langs: str | None = None) -> list[OCRLine]:
    settings = get_settings()
    langs_key = langs or settings.default_ocr_langs
    reader = _build_paddleocr_reader(langs_key)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    result = reader.predict(image_array)
    if not result:
        return []

    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    polys = page.get("dt_polys", []) or page.get("rec_polys", [])

    rows: list[dict] = []
    for text, score, poly in zip(texts, scores, polys):
        text = str(text).strip()
        if not text:
            continue
        arr = np.asarray(poly)
        if arr.size == 0:
            continue
        xs = arr[:, 0].astype(float)
        ys = arr[:, 1].astype(float)
        rows.append(
            {
                "text": text,
                "ocr_confidence": float(score),
                "bbox_x1": float(xs.min()),
                "bbox_y1": float(ys.min()),
                "bbox_x2": float(xs.max()),
                "bbox_y2": float(ys.max()),
            }
        )
    return _with_line_order(rows)


def ocr_backend_is_available(backend: str) -> bool:
    name = (backend or "").strip().lower()
    try:
        if name == "paddleocr":
            _configure_paddle_env()
            import paddleocr  # noqa: F401
            return True
        if name == "easyocr":
            import easyocr  # noqa: F401
            return True
    except Exception:
        return False
    return False


def get_ocr_runtime_info() -> dict:
    settings = get_settings()
    return {
        "requested_backend": settings.ocr_backend,
        "fallback_backend": settings.ocr_fallback_backend,
        "available_backends": {
            "paddleocr": ocr_backend_is_available("paddleocr"),
            "easyocr": ocr_backend_is_available("easyocr"),
        },
        "paddle_cache_dir": settings.paddle_cache_dir,
    }


def run_ocr(
    image_bytes: bytes,
    langs: str | None = None,
    backend: str | None = None,
) -> OCRRunResult:
    settings = get_settings()
    requested = (backend or settings.ocr_backend or "easyocr").strip().lower()
    fallback = (settings.ocr_fallback_backend or "").strip().lower()

    backends = [requested]
    if fallback and fallback not in backends:
        backends.append(fallback)

    errors: list[str] = []
    for name in backends:
        try:
            if name == "paddleocr":
                return OCRRunResult(backend="paddleocr", lines=run_paddleocr(image_bytes, langs=langs))
            if name == "easyocr":
                return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
            errors.append(f"{name}: unsupported backend")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("All OCR backends failed. " + " | ".join(errors))


def extract_texts(ocr_lines: Iterable[OCRLine]) -> list[str]:
    return [line.text for line in ocr_lines if line.text.strip()]

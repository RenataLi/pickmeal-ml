from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

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

PADDLE_REC_MODEL_NAMES = {
    "en": "en_PP-OCRv5_mobile_rec",
    "ru": "cyrillic_PP-OCRv5_mobile_rec",
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
    ordered_rows = _sort_rows_for_reading(rows)
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
        for idx, row in enumerate(ordered_rows, start=1)
    ]


def _cluster_key(value: float, tolerance: float) -> int:
    if tolerance <= 0:
        return 0
    return int(round(value / tolerance))


def _sort_rows_for_reading(rows: list[dict]) -> list[dict]:
    if len(rows) <= 1:
        return rows
    if any(row.get("bbox_x1") is None or row.get("bbox_y1") is None for row in rows):
        return rows

    min_x = min(float(row["bbox_x1"]) for row in rows)
    max_x = max(float(row.get("bbox_x2") or row["bbox_x1"]) for row in rows)
    page_width = max(1.0, max_x - min_x)
    tolerance = min(max(page_width * 0.18, 80.0), 260.0)

    return sorted(
        rows,
        key=lambda row: (
            _cluster_key(float(row["bbox_x1"]) - min_x, tolerance),
            float(row["bbox_y1"]),
            float(row["bbox_x1"]),
        ),
    )


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


def _load_image_array(image_bytes: bytes, max_side: int) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_side:
        scale = max_side / float(longest_side)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(image)


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    return image.size


def _resolve_paddle_lang(langs_key: str) -> str:
    langs = _normalize_langs(langs_key)
    for lang in langs:
        if lang in PADDLE_LANG_ALIASES:
            return PADDLE_LANG_ALIASES[lang]
    return "en"


def _resolve_paddle_rec_model(langs_key: str) -> str:
    return PADDLE_REC_MODEL_NAMES.get(_resolve_paddle_lang(langs_key), "en_PP-OCRv5_mobile_rec")


def _paddle_model_dir(model_name: str) -> str:
    settings = get_settings()
    return os.path.join(settings.paddle_cache_dir, "official_models", model_name)


def paddle_models_are_cached(langs_key: str) -> bool:
    det_dir = _paddle_model_dir(get_settings().paddle_text_detection_model_name)
    rec_dir = _paddle_model_dir(_resolve_paddle_rec_model(langs_key))
    return os.path.isdir(det_dir) and os.path.isdir(rec_dir)


def _accept_easy_result(lines: list[OCRLine], image_bytes: bytes) -> bool:
    width, height = _image_size(image_bytes)
    area = width * height
    mean_conf = 0.0
    confs = [float(line.ocr_confidence) for line in lines if line.ocr_confidence is not None]
    if confs:
        mean_conf = sum(confs) / len(confs)

    if len(lines) >= 12 and mean_conf >= 0.55:
        return True
    if len(lines) >= 8 and mean_conf >= 0.68:
        return True
    if area >= 2_000_000 and len(lines) >= 8:
        return True
    return False


def _prefer_easy_for_image(image_bytes: bytes) -> bool:
    width, height = _image_size(image_bytes)
    longest_side = max(width, height)
    area = width * height
    return longest_side > 2200 or area > 2_200_000


@lru_cache(maxsize=4)
def _build_paddleocr_reader(langs_key: str):
    _configure_paddle_env()
    settings = get_settings()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "paddleocr is not installed. Install paddleocr and paddlepaddle or switch OCR backend to easyocr."
        ) from exc

    return PaddleOCR(
        text_detection_model_name=settings.paddle_text_detection_model_name,
        text_recognition_model_name=_resolve_paddle_rec_model(langs_key),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=settings.ocr_max_image_side,
    )


def run_easyocr(image_bytes: bytes, langs: str | None = None) -> list[OCRLine]:
    settings = get_settings()
    langs_key = langs or settings.default_ocr_langs
    reader = _build_reader(langs_key)

    image_array = _load_image_array(image_bytes, settings.ocr_max_image_side)
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

    image_array = _load_image_array(image_bytes, settings.ocr_max_image_side)
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
        "paddle_detection_model": settings.paddle_text_detection_model_name,
        "ocr_max_image_side": settings.ocr_max_image_side,
    }


def run_ocr(
    image_bytes: bytes,
    langs: str | None = None,
    backend: str | None = None,
) -> OCRRunResult:
    settings = get_settings()
    requested = (backend or settings.ocr_backend or "easyocr").strip().lower()
    fallback = (settings.ocr_fallback_backend or "").strip().lower()
    easy_available = ocr_backend_is_available("easyocr")
    paddle_available = ocr_backend_is_available("paddleocr")

    if requested == "auto":
        langs_key = langs or settings.default_ocr_langs
        if easy_available and (_prefer_easy_for_image(image_bytes) or not paddle_available):
            return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
        if paddle_available and paddle_models_are_cached(langs_key):
            try:
                return OCRRunResult(backend="paddleocr", lines=run_paddleocr(image_bytes, langs=langs))
            except Exception:
                if easy_available:
                    easy_lines = run_easyocr(image_bytes, langs=langs)
                    return OCRRunResult(backend="easyocr", lines=easy_lines)
                raise
        if easy_available:
            return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
        if paddle_available:
            return OCRRunResult(backend="paddleocr", lines=run_paddleocr(image_bytes, langs=langs))
        raise RuntimeError("No OCR backend is available in the current environment.")

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

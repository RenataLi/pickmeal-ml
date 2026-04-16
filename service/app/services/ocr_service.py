from __future__ import annotations

import io
import os
import platform
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

PADDLE_BACKEND_ALIASES = {
    "paddleocr": "paddleocr_quality",
    "paddleocr_quality": "paddleocr_quality",
    "paddleocr_mobile": "paddleocr_mobile",
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


@lru_cache(maxsize=2)
def _build_rapidocr_engine():
    try:
        from rapidocr import RapidOCR
    except Exception as exc:
        raise RuntimeError(
            "rapidocr is not installed. Install rapidocr and onnxruntime or switch OCR backend."
        ) from exc
    return RapidOCR()


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


def _paddle_runtime_disabled_reason() -> str | None:
    allow_override = os.getenv("PICKMEAL_ALLOW_PADDLE_DOCKER_ARM", "").strip().lower() in {"1", "true", "yes", "on"}
    if allow_override:
        return None
    if not os.path.exists("/.dockerenv"):
        return None
    if platform.system().lower() != "linux":
        return None
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "paddleocr is disabled in Docker on Linux ARM because this runtime is unstable and may segfault"
    return None


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


def _resolve_paddle_detector_model(backend_name: str) -> str:
    settings = get_settings()
    normalized = PADDLE_BACKEND_ALIASES.get((backend_name or "").strip().lower(), "paddleocr_quality")
    if normalized == "paddleocr_mobile":
        return settings.paddle_text_detection_mobile_model_name
    return settings.paddle_text_detection_model_name


def _paddle_model_dir(model_name: str) -> str:
    settings = get_settings()
    return os.path.join(settings.paddle_cache_dir, "official_models", model_name)


def paddle_models_are_cached(langs_key: str, backend_name: str = "paddleocr_quality") -> bool:
    det_dir = _paddle_model_dir(_resolve_paddle_detector_model(backend_name))
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


@lru_cache(maxsize=8)
def _build_paddleocr_reader(langs_key: str, detector_model_name: str):
    _configure_paddle_env()
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError(
            "paddleocr is not installed. Install paddleocr and paddlepaddle or switch OCR backend to easyocr."
        ) from exc

    return PaddleOCR(
        text_detection_model_name=detector_model_name,
        text_recognition_model_name=_resolve_paddle_rec_model(langs_key),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=get_settings().ocr_max_image_side,
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


def run_rapidocr(image_bytes: bytes, langs: str | None = None) -> list[OCRLine]:
    settings = get_settings()
    engine = _build_rapidocr_engine()
    image_array = _load_image_array(image_bytes, settings.ocr_max_image_side)
    result = engine(image_array)
    if result is None:
        return []
    if isinstance(result, tuple):
        result = result[0] if result else None
        if result is None:
            return []

    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    if boxes is None:
        boxes = []
    if texts is None:
        texts = []
    if scores is None:
        scores = []

    rows: list[dict] = []
    for idx, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        score = None
        if idx < len(scores):
            try:
                score = float(scores[idx])
            except Exception:
                score = None
        if idx < len(boxes):
            arr = np.asarray(boxes[idx])
            if arr.size > 0:
                xs = arr[:, 0].astype(float)
                ys = arr[:, 1].astype(float)
                rows.append(
                    {
                        "text": text,
                        "ocr_confidence": score,
                        "bbox_x1": float(xs.min()),
                        "bbox_y1": float(ys.min()),
                        "bbox_x2": float(xs.max()),
                        "bbox_y2": float(ys.max()),
                    }
                )
                continue
        rows.append({"text": text, "ocr_confidence": score})
    return _with_line_order(rows)


def run_paddleocr(image_bytes: bytes, langs: str | None = None, backend_name: str = "paddleocr_quality") -> list[OCRLine]:
    settings = get_settings()
    langs_key = langs or settings.default_ocr_langs
    detector_model_name = _resolve_paddle_detector_model(backend_name)
    reader = _build_paddleocr_reader(langs_key, detector_model_name)

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
        if name in PADDLE_BACKEND_ALIASES:
            if _paddle_runtime_disabled_reason():
                return False
            _configure_paddle_env()
            import paddleocr  # noqa: F401
            return True
        if name == "rapidocr":
            import rapidocr  # noqa: F401
            import onnxruntime  # noqa: F401
            return True
        if name == "easyocr":
            import easyocr  # noqa: F401
            return True
    except Exception:
        return False
    return False


def get_ocr_runtime_info() -> dict:
    settings = get_settings()
    paddle_disabled_reason = _paddle_runtime_disabled_reason()
    return {
        "requested_backend": settings.ocr_backend,
        "fallback_backend": settings.ocr_fallback_backend,
        "available_backends": {
            "paddleocr_quality": ocr_backend_is_available("paddleocr_quality"),
            "paddleocr_mobile": ocr_backend_is_available("paddleocr_mobile"),
            "rapidocr": ocr_backend_is_available("rapidocr"),
            "easyocr": ocr_backend_is_available("easyocr"),
        },
        "paddle_cache_dir": settings.paddle_cache_dir,
        "paddle_detection_model": settings.paddle_text_detection_model_name,
        "paddle_mobile_detection_model": settings.paddle_text_detection_mobile_model_name,
        "ocr_max_image_side": settings.ocr_max_image_side,
        "paddle_disabled_reason": paddle_disabled_reason,
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
    rapid_available = ocr_backend_is_available("rapidocr")
    paddle_available = ocr_backend_is_available("paddleocr_quality")

    if requested == "auto":
        langs_key = langs or settings.default_ocr_langs
        if rapid_available:
            return OCRRunResult(backend="rapidocr", lines=run_rapidocr(image_bytes, langs=langs))
        if easy_available and (_prefer_easy_for_image(image_bytes) or not paddle_available):
            return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
        if paddle_available and paddle_models_are_cached(langs_key, backend_name="paddleocr_mobile"):
            try:
                return OCRRunResult(backend="paddleocr_mobile", lines=run_paddleocr(image_bytes, langs=langs, backend_name="paddleocr_mobile"))
            except Exception:
                if easy_available:
                    easy_lines = run_easyocr(image_bytes, langs=langs)
                    return OCRRunResult(backend="easyocr", lines=easy_lines)
                raise
        if easy_available:
            return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
        if rapid_available:
            return OCRRunResult(backend="rapidocr", lines=run_rapidocr(image_bytes, langs=langs))
        if paddle_available:
            return OCRRunResult(backend="paddleocr_mobile", lines=run_paddleocr(image_bytes, langs=langs, backend_name="paddleocr_mobile"))
        raise RuntimeError("No OCR backend is available in the current environment.")

    backends = [requested]
    if fallback and fallback not in backends:
        backends.append(fallback)

    errors: list[str] = []
    for name in backends:
        try:
            if name in PADDLE_BACKEND_ALIASES:
                reason = _paddle_runtime_disabled_reason()
                if reason:
                    errors.append(f"{name}: {reason}")
                    continue
                normalized = PADDLE_BACKEND_ALIASES[name]
                return OCRRunResult(backend=normalized, lines=run_paddleocr(image_bytes, langs=langs, backend_name=normalized))
            if name == "rapidocr":
                if not ocr_backend_is_available("rapidocr"):
                    errors.append(f"{name}: backend unavailable")
                    continue
                return OCRRunResult(backend="rapidocr", lines=run_rapidocr(image_bytes, langs=langs))
            if name == "easyocr":
                if not ocr_backend_is_available("easyocr"):
                    errors.append(f"{name}: backend unavailable")
                    continue
                return OCRRunResult(backend="easyocr", lines=run_easyocr(image_bytes, langs=langs))
            errors.append(f"{name}: unsupported backend")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("All OCR backends failed. " + " | ".join(errors))


def extract_texts(ocr_lines: Iterable[OCRLine]) -> list[str]:
    return [line.text for line in ocr_lines if line.text.strip()]

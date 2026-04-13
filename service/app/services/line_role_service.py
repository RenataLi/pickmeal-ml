from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import joblib
import pandas as pd

from ..config import get_settings
from ..schemas import LineRolePrediction, OCRLine

PRICE_ONLY_RE = re.compile(
    r"^[€$£₹₽]?\s*\d{1,5}(?:[.,]\d{1,2})?\s*(?:rub|usd|eur|inr|€|\$|£|₹|₽)?$",
    re.IGNORECASE,
)

NUMERIC_COLS = [
    "word_count",
    "char_count",
    "digit_count",
    "uppercase_ratio",
    "title_ratio",
    "comma_count",
    "has_price_token",
    "price_only_flag",
    "ocr_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_center_x",
    "bbox_center_y",
    "line_order",
]


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    return " ".join(text.strip().split())


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def uppercase_ratio(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def title_ratio(text: str) -> float:
    tokens = [t for t in normalize_text(text).split() if any(ch.isalpha() for ch in t)]
    if not tokens:
        return 0.0
    good = 0
    for tok in tokens:
        clean = re.sub(r"[^A-Za-z]+", "", tok)
        if clean and clean[0].isupper():
            good += 1
    return good / len(tokens)


def looks_like_price_only(text: str) -> bool:
    return bool(PRICE_ONLY_RE.match(normalize_text(text)))


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def line_to_features(line: OCRLine) -> dict:
    text = normalize_text(line.text)
    x1 = safe_float(line.bbox_x1)
    y1 = safe_float(line.bbox_y1)
    x2 = safe_float(line.bbox_x2)
    y2 = safe_float(line.bbox_y2)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = (x1 + x2) / 2.0 if x1 or x2 else 0.0
    center_y = (y1 + y2) / 2.0 if y1 or y2 else 0.0

    return {
        "text": text,
        "word_count": word_count(text),
        "char_count": len(text),
        "digit_count": sum(ch.isdigit() for ch in text),
        "uppercase_ratio": round(uppercase_ratio(text), 6),
        "title_ratio": round(title_ratio(text), 6),
        "comma_count": text.count(","),
        "has_price_token": int(any(ch.isdigit() for ch in text)),
        "price_only_flag": int(looks_like_price_only(text)),
        "ocr_confidence": safe_float(line.ocr_confidence),
        "bbox_width": round(width, 4),
        "bbox_height": round(height, 4),
        "bbox_center_x": round(center_x, 4),
        "bbox_center_y": round(center_y, 4),
        "line_order": int(line.line_order or 0),
    }


@lru_cache(maxsize=1)
def load_model():
    settings = get_settings()
    model_path = settings.project_root / settings.line_role_model_path
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def line_role_model_is_available() -> bool:
    settings = get_settings()
    model_path = settings.project_root / settings.line_role_model_path
    return model_path.exists()


def predict_line_roles(lines: Iterable[OCRLine]) -> list[LineRolePrediction]:
    model = load_model()
    if model is None:
        return []

    line_list = list(lines)
    if not line_list:
        return []

    df = pd.DataFrame([line_to_features(line) for line in line_list])
    features = df[["text"] + NUMERIC_COLS]
    pred_labels = model.predict(features)

    pred_scores = None
    if hasattr(model, "predict_proba"):
        try:
            pred_scores = model.predict_proba(features)
        except Exception:
            pred_scores = None

    result: list[LineRolePrediction] = []
    for idx, (line, label) in enumerate(zip(line_list, pred_labels)):
        score = None
        if pred_scores is not None:
            try:
                score = float(max(pred_scores[idx]))
            except Exception:
                score = None

        result.append(
            LineRolePrediction(
                text=normalize_text(line.text),
                predicted_label=str(label),
                predicted_score=score,
                line_order=line.line_order,
                ocr_confidence=line.ocr_confidence,
                bbox_x1=line.bbox_x1,
                bbox_y1=line.bbox_y1,
                bbox_x2=line.bbox_x2,
                bbox_y2=line.bbox_y2,
            )
        )
    return result

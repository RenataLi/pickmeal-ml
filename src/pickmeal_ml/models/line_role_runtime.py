from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


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


@dataclass
class LineRolePrediction:
    label: str
    score: float | None = None


def resolve_project_root() -> Path:
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    if (root.parent / "src").exists():
        return root.parent
    return Path(__file__).resolve().parents[3]


def get_default_model_path() -> Path:
    rel_path = os.getenv(
        "PICKMEAL_LINE_ROLE_MODEL_PATH",
        "reports/line_role_expanded_sgd_v1/line_role_logreg.joblib",
    )
    return resolve_project_root() / rel_path


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


def line_to_features(line) -> dict:
    text = normalize_text(getattr(line, "text", None) if not isinstance(line, dict) else line.get("text"))
    x1 = safe_float(getattr(line, "bbox_x1", None) if not isinstance(line, dict) else line.get("bbox_x1"))
    y1 = safe_float(getattr(line, "bbox_y1", None) if not isinstance(line, dict) else line.get("bbox_y1"))
    x2 = safe_float(getattr(line, "bbox_x2", None) if not isinstance(line, dict) else line.get("bbox_x2"))
    y2 = safe_float(getattr(line, "bbox_y2", None) if not isinstance(line, dict) else line.get("bbox_y2"))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    center_x = (x1 + x2) / 2.0 if x1 or x2 else 0.0
    center_y = (y1 + y2) / 2.0 if y1 or y2 else 0.0
    line_order = getattr(line, "line_order", None) if not isinstance(line, dict) else line.get("line_order")
    ocr_confidence = getattr(line, "ocr_confidence", None) if not isinstance(line, dict) else line.get("ocr_confidence")

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
        "ocr_confidence": safe_float(ocr_confidence),
        "bbox_width": round(width, 4),
        "bbox_height": round(height, 4),
        "bbox_center_x": round(center_x, 4),
        "bbox_center_y": round(center_y, 4),
        "line_order": int(line_order or 0),
    }


@lru_cache(maxsize=1)
def load_line_role_model(model_path: str | None = None):
    path = Path(model_path) if model_path is not None else get_default_model_path()
    if not path.exists():
        return None
    return joblib.load(path)


def predict_line_roles(lines: Iterable, model_path: str | None = None) -> list[LineRolePrediction]:
    model = load_line_role_model(model_path)
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

    rows = []
    for idx, label in enumerate(pred_labels):
        score = None
        if pred_scores is not None:
            try:
                score = float(max(pred_scores[idx]))
            except Exception:
                score = None
        rows.append(LineRolePrediction(label=str(label), score=score))
    return rows

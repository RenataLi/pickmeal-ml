from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get_settings


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_summary() -> dict:
    settings = get_settings()
    summary_path = settings.project_root / settings.dataset_summary_path
    return _load_json(summary_path)


def load_parser_metrics() -> dict:
    settings = get_settings()
    metrics_dir = settings.project_root / settings.parser_metrics_dir
    result = {
        "valid": _load_json(metrics_dir / "parser_metrics_valid.json"),
        "test": _load_json(metrics_dir / "parser_metrics_test.json"),
    }
    return result


def load_parser_comparison() -> list[dict]:
    settings = get_settings()
    candidates = [
        settings.project_root / "reports" / "parser_comparison" / "parser_v1_vs_v2.csv",
        settings.project_root / "reports" / "parser_comparison" / "parser_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
    return []

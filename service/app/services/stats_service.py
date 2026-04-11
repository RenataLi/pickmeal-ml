from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import get_settings


PARSER_REPORTS = [
    ("Parser v1", "reports/parser_baseline"),
    ("Parser v2", "reports/parser_baseline_v2"),
    ("Parser line-role v3", "reports/parser_line_role_v3"),
    ("Parser line-role v3 layout", "reports/parser_line_role_v3_layout"),
    ("Parser hybrid merge v31", "reports/parser_hybrid_merge_v31"),
    ("Parser hybrid v4 layout", "reports/parser_hybrid_v4_layout"),
]


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
    rows: list[dict] = []
    active_dir = settings.parser_metrics_dir
    for label, rel_dir in PARSER_REPORTS:
        metrics_dir = settings.project_root / rel_dir
        valid = _load_json(metrics_dir / "parser_metrics_valid.json")
        test = _load_json(metrics_dir / "parser_metrics_test.json")
        if valid:
            rows.append(
                {
                    "model": label,
                    "split": "valid",
                    "metrics_dir": rel_dir,
                    "is_active": rel_dir == active_dir,
                    **valid,
                }
            )
        if test:
            rows.append(
                {
                    "model": label,
                    "split": "test",
                    "metrics_dir": rel_dir,
                    "is_active": rel_dir == active_dir,
                    **test,
                }
            )
    return rows


def load_line_role_metrics() -> dict:
    settings = get_settings()
    path = settings.project_root / settings.line_role_metrics_path
    return _load_json(path)


def load_line_role_report() -> dict:
    settings = get_settings()
    path = settings.project_root / settings.line_role_report_path
    return _load_json(path)

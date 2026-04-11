from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel

from .bootstrap import ensure_src_on_path


class Settings(BaseModel):
    project_root: Path
    api_title: str = "PickMeal Service"
    api_version: str = "0.1.0"
    parser_module: str = os.getenv(
        "PICKMEAL_PARSER_MODULE",
        "pickmeal_ml.models.menu_structuring_baseline_v2",
    )
    default_ocr_langs: str = os.getenv("PICKMEAL_OCR_LANGS", "en")
    parser_metrics_dir: str = os.getenv("PICKMEAL_PARSER_METRICS_DIR", "reports/parser_baseline_v2")
    dataset_summary_path: str = os.getenv("PICKMEAL_DATASET_SUMMARY_PATH", "data/interim/dataset_summary.json")
    line_role_model_path: str = os.getenv("PICKMEAL_LINE_ROLE_MODEL_PATH", "reports/line_role_baseline/line_role_logreg.joblib")
    line_role_metrics_path: str = os.getenv("PICKMEAL_LINE_ROLE_METRICS_PATH", "reports/line_role_baseline/line_role_metrics.json")
    line_role_report_path: str = os.getenv("PICKMEAL_LINE_ROLE_REPORT_PATH", "reports/line_role_baseline/line_role_classification_report.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = ensure_src_on_path()
    return Settings(project_root=project_root)

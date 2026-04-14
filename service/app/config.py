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
    database_url: str | None = os.getenv("PICKMEAL_DATABASE_URL") or None
    embedding_dimensions: int = int(os.getenv("PICKMEAL_EMBEDDING_DIMENSIONS", "256"))
    embedding_model_name: str = os.getenv("PICKMEAL_EMBEDDING_MODEL", "hashing_v1")
    parser_module: str = os.getenv(
        "PICKMEAL_PARSER_MODULE",
        "pickmeal_ml.models.menu_structuring_cascade_v1",
    )
    default_ocr_langs: str = os.getenv("PICKMEAL_OCR_LANGS", "en")
    ocr_backend: str = os.getenv("PICKMEAL_OCR_BACKEND", "auto")
    ocr_fallback_backend: str = os.getenv("PICKMEAL_OCR_FALLBACK_BACKEND", "easyocr")
    paddle_cache_dir: str
    paddle_mpl_config_dir: str
    paddle_text_detection_model_name: str = os.getenv("PICKMEAL_PADDLE_DET_MODEL", "PP-OCRv5_server_det")
    ocr_max_image_side: int = int(os.getenv("PICKMEAL_OCR_MAX_IMAGE_SIDE", "2560"))
    parser_metrics_dir: str = os.getenv("PICKMEAL_PARSER_METRICS_DIR", "reports/parser_cascade_v1_expanded")
    dataset_summary_path: str = os.getenv("PICKMEAL_DATASET_SUMMARY_PATH", "data/interim/dataset_summary.json")
    line_role_model_path: str = os.getenv("PICKMEAL_LINE_ROLE_MODEL_PATH", "reports/line_role_expanded_sgd_v1/line_role_logreg.joblib")
    line_role_metrics_path: str = os.getenv("PICKMEAL_LINE_ROLE_METRICS_PATH", "reports/line_role_expanded_sgd_v1/line_role_metrics.json")
    line_role_report_path: str = os.getenv("PICKMEAL_LINE_ROLE_REPORT_PATH", "reports/line_role_expanded_sgd_v1/line_role_classification_report.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = ensure_src_on_path()
    cache_root = project_root / ".cache"
    return Settings(
        project_root=project_root,
        paddle_cache_dir=os.getenv("PICKMEAL_PADDLE_CACHE_DIR", str(cache_root / "paddlex_cache")),
        paddle_mpl_config_dir=os.getenv("PICKMEAL_PADDLE_MPLCONFIGDIR", str(cache_root / "matplotlib_cache")),
    )

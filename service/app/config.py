from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel

from .bootstrap import ensure_src_on_path


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


class Settings(BaseModel):
    project_root: Path
    api_title: str = "PickMeal Service"
    api_version: str = "0.1.0"
    database_url: str | None = (
        os.getenv("PICKMEAL_DATABASE_URL")
        or os.getenv("PICKMEAL_DATABASE_URL_LOCAL")
        or "postgresql://pickmeal:pickmeal@localhost:5432/pickmeal"
    )
    embedding_dimensions: int = int(os.getenv("PICKMEAL_EMBEDDING_DIMENSIONS", "256"))
    embedding_model_name: str = os.getenv("PICKMEAL_EMBEDDING_MODEL", "hashing_v1")
    llm_enabled: bool = os.getenv("PICKMEAL_LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    llm_base_url: str | None = os.getenv("PICKMEAL_LLM_BASE_URL") or None
    llm_api_key: str | None = os.getenv("PICKMEAL_LLM_API_KEY") or None
    llm_model: str | None = os.getenv("PICKMEAL_LLM_MODEL") or None
    llm_timeout_seconds: int = int(os.getenv("PICKMEAL_LLM_TIMEOUT_SECONDS", "45"))
    llm_max_items_per_request: int = int(os.getenv("PICKMEAL_LLM_MAX_ITEMS_PER_REQUEST", "6"))
    rag_enabled: bool = os.getenv("PICKMEAL_RAG_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    rag_top_k: int = int(os.getenv("PICKMEAL_RAG_TOP_K", "4"))
    rag_dataset_path: str = os.getenv("PICKMEAL_RAG_DATASET_PATH", "data/processed/gold/menu_gold_all_batches.csv")
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
    paddle_text_detection_mobile_model_name: str = os.getenv("PICKMEAL_PADDLE_DET_MODEL_MOBILE", "PP-OCRv5_mobile_det")
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

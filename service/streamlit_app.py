from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def normalize_api_url(value: str | None) -> str:
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return "http://api:8000" if running_in_docker() else "http://127.0.0.1:8000"
    if not running_in_docker() and candidate in {"http://api:8000", "http://api"}:
        return "http://127.0.0.1:8000"
    return candidate


def resolve_api_url() -> str:
    if running_in_docker():
        return normalize_api_url(
            os.getenv("PICKMEAL_API_URL")
            or os.getenv("PICKMEAL_API_URL_DOCKER")
            or "http://api:8000"
        )
    return normalize_api_url(os.getenv("PICKMEAL_API_URL") or "http://127.0.0.1:8000")


API_URL = resolve_api_url()
DEFAULT_OCR_BACKEND = (os.getenv("PICKMEAL_OCR_BACKEND") or "easyocr").strip().lower()
OCR_BACKEND_OPTIONS = ["rapidocr", "easyocr", "paddleocr_mobile", "paddleocr_quality", "auto"]
OCR_BACKEND_LABELS = {
    "rapidocr": "RapidOCR",
    "easyocr": "EasyOCR",
    "paddleocr_mobile": "PaddleOCR Mobile",
    "paddleocr_quality": "PaddleOCR Quality",
    "auto": "Auto",
}


def button_stretch(label: str, **kwargs):
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        try:
            return st.button(label, use_container_width=True, **kwargs)
        except TypeError:
            return st.button(label, **kwargs)


def show_image(image, caption: str | None = None):
    try:
        st.image(image, caption=caption, width="stretch")
    except TypeError:
        try:
            st.image(image, caption=caption, use_container_width=True)
        except TypeError:
            st.image(image, caption=caption, use_column_width=True)


def show_dataframe(df: pd.DataFrame, height: int | None = None):
    try:
        st.dataframe(df, width="stretch", hide_index=True, height=height)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True, height=height)
        except TypeError:
            st.dataframe(df, height=height)


def load_preview_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    if angle % 360 == 0:
        return image
    return image.rotate(-angle, expand=True)


def image_to_upload_bytes(image: Image.Image, original_name: str) -> tuple[bytes, str]:
    suffix = ".png"
    lowered = original_name.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        suffix = ".jpg"
    buf = io.BytesIO()
    if suffix == ".jpg":
        image.save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "image/png"
    return buf.getvalue(), mime


def fit_image_for_display(image: Image.Image, max_height: int = 760, max_width: int = 1100) -> Image.Image:
    canvas = image.copy()
    if canvas.height <= max_height and canvas.width <= max_width:
        return canvas
    canvas.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return canvas


def render_preview_frame(image: Image.Image, caption: str | None = None, frame_height: int = 720):
    prepared = fit_image_for_display(image, max_height=frame_height, max_width=1200)
    buf = io.BytesIO()
    prepared.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    caption_html = f"<div class='pm-preview-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class="pm-preview-frame">
            <div class="pm-preview-shell" style="height:{frame_height}px;">
                <img src="data:image/png;base64,{encoded}" alt="Menu preview" />
            </div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    defaults = {
        "api_url_input": API_URL,
        "parsed_payload": None,
        "parse_error": None,
        "recommend_payload": None,
        "recommend_error": None,
        "llm_payload": None,
        "llm_error": None,
        "parsed_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["api_url_input"] = normalize_api_url(st.session_state.get("api_url_input"))


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def format_api_error(error) -> str:
    if error is None:
        return ""
    if isinstance(error, dict):
        detail = error.get("detail")
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict):
                    loc = item.get("loc")
                    msg = item.get("msg")
                    if loc and msg:
                        parts.append(f"{'.'.join(str(x) for x in loc)}: {msg}")
                    elif msg:
                        parts.append(str(msg))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return " ".join(part for part in parts if part).strip()
        if detail:
            return str(detail)
        return json.dumps(error, ensure_ascii=False)
    return str(error)


def fetch_api_json(api_url: str, path: str, method: str = "GET", json_payload: dict | None = None, timeout: int = 30) -> dict | list | None:
    try:
        if method.upper() == "POST":
            response = requests.post(f"{api_url}{path}", json=json_payload, timeout=timeout)
        else:
            response = requests.get(f"{api_url}{path}", timeout=timeout)
        if response.ok:
            return response.json()
    except Exception:
        return None
    return None


def load_parser_metric_rows() -> pd.DataFrame:
    parser_dirs = [
        ("Baseline v1", "parser_baseline"),
        ("Baseline v2", "parser_baseline_v2"),
        ("Baseline v2 expanded", "parser_baseline_v2_expanded"),
        ("Baseline v3 layout", "parser_baseline_v3_layout_expanded"),
        ("Cascade v1 expanded", "parser_cascade_v1_expanded"),
        ("Line-role hybrid v2", "parser_line_role_hybrid_v2_expanded"),
        ("Line-role v3", "parser_line_role_v3"),
        ("Line-role v3 layout", "parser_line_role_v3_layout"),
        ("Hybrid merge v31", "parser_hybrid_merge_v31"),
        ("Hybrid v4 layout", "parser_hybrid_v4_layout"),
    ]
    rows: list[dict] = []
    for label, rel_dir in parser_dirs:
        for split in ["valid", "test"]:
            path = PROJECT_ROOT / "reports" / rel_dir / f"parser_metrics_{split}.json"
            metrics = read_json_if_exists(path) or {}
            if metrics:
                rows.append(
                    {
                        "model": label,
                        "split": split,
                        "metrics_dir": rel_dir,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def load_line_role_metric_rows() -> pd.DataFrame:
    rows = []
    candidates = [
        ("Line-role baseline", "line_role_baseline"),
        ("Line-role expanded v1", "line_role_expanded_v1"),
        ("Line-role expanded SGD v1", "line_role_expanded_sgd_v1"),
        ("Line-role contextual CatBoost v1", "line_role_contextual_catboost_v1"),
    ]
    for label, rel_dir in candidates:
        metrics = read_json_if_exists(PROJECT_ROOT / "reports" / rel_dir / "line_role_metrics.json") or {}
        if metrics:
            rows.append({"model": label, "metrics_dir": rel_dir, **metrics})
    return pd.DataFrame(rows)


def load_ocr_metric_rows() -> pd.DataFrame:
    benchmark_labels = {
        "ocr_backend_benchmark_v1": "OCR benchmark v1",
        "ocr_backend_benchmark_v2": "OCR benchmark v2",
        "ocr_backend_benchmark_v3": "OCR benchmark v3",
        "ocr_backend_benchmark_cascade_v1": "OCR benchmark cascade v1",
    }
    rows: list[dict] = []
    for rel_dir, run_label in benchmark_labels.items():
        metrics = read_json_if_exists(PROJECT_ROOT / "reports" / rel_dir / "ocr_backend_metrics.json") or []
        for row in metrics:
            rows.append({"benchmark_run": run_label, "metrics_dir": rel_dir, **row})
    return pd.DataFrame(rows)


def normalize_metrics_dir(value: str | None) -> str:
    if not value:
        return ""
    return Path(str(value)).name


def merge_metric_frames(*frames: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    usable_frames = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not usable_frames:
        return pd.DataFrame()
    merged = pd.concat(usable_frames, ignore_index=True)
    if "metrics_dir" in merged.columns:
        merged["metrics_dir"] = merged["metrics_dir"].map(normalize_metrics_dir)
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="first")
    return merged


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --pm-bg: #f7f3ed;
            --pm-surface: #ffffff;
            --pm-card: rgba(255, 255, 255, 0.90);
            --pm-ink: #1f2933;
            --pm-muted: #5f6875;
            --pm-line: rgba(31, 41, 51, 0.12);
            --pm-accent: #7a695c;
            --pm-accent-mid: #a08c7f;
            --pm-accent-dark: #c98a6a;
            --pm-accent-soft: #f1e2d7;
            --pm-green: #315b41;
            --pm-blue: #2f6f84;
            --pm-shadow: 0 18px 48px rgba(63, 43, 28, 0.10);
            --pm-radius-lg: 26px;
            --pm-radius-md: 18px;
        }

        .stApp {
            background:
                radial-gradient(980px 320px at 22% 6%, rgba(255,255,255,0.74), transparent 66%),
                radial-gradient(860px 320px at 82% 10%, rgba(255,255,255,0.30), transparent 68%),
                radial-gradient(940px 340px at 92% 0%, rgba(201,138,106,0.16), transparent 52%),
                radial-gradient(1040px 360px at 8% -12%, rgba(111,126,93,0.18), transparent 54%),
                radial-gradient(720px 260px at 14% 78%, rgba(111,126,93,0.08), transparent 68%),
                #f7f3ed;
            color: var(--pm-ink);
            font-family: "Avenir Next", "Helvetica Neue", sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid var(--pm-line);
            border-radius: 999px;
            padding: 0.35rem;
            width: fit-content;
            box-shadow: 0 18px 38px rgba(50, 39, 21, 0.12), inset 0 1px 0 rgba(255,255,255,0.5);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.7rem 1rem;
            color: var(--pm-muted);
            font-weight: 600;
            height: auto;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6f7e5d 0%, #9aa08a 52%, #c98a6a 100%);
            color: #fefaf3 !important;
            border: 1px solid rgba(255,255,255,0.32);
            box-shadow: 0 16px 32px rgba(122, 105, 92, 0.24), inset 0 1px 0 rgba(255,255,255,0.24);
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background: transparent !important;
            height: 0 !important;
        }

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.22);
            background: linear-gradient(135deg, var(--pm-accent) 0%, var(--pm-accent-mid) 50%, var(--pm-accent-dark) 100%);
            color: #fff8ef;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 22px 42px rgba(122, 105, 92, 0.28), inset 0 1px 0 rgba(255,255,255,0.24);
        }

        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
            filter: brightness(1.045);
            transform: translateY(-1px);
        }

        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {
            border-radius: 16px !important;
            border-color: rgba(18, 32, 39, 0.14) !important;
            background: rgba(255,255,255,0.94) !important;
            box-shadow: 0 1px 0 rgba(255,255,255,0.85), 0 14px 28px rgba(50, 39, 21, 0.08) !important;
        }

        .stFileUploader {
            background: rgba(255,255,255,0.74);
            border: 1px dashed rgba(18, 32, 39, 0.18);
            border-radius: 22px;
            padding: 0.5rem 0.8rem;
            box-shadow: 0 18px 34px rgba(50, 39, 21, 0.10);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0f5f6 0%, #faf2eb 100%);
            border-right: 1px solid #d7cdc2;
        }

        [data-testid="stSidebar"] * {
            color: #21323b;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(18, 32, 39, 0.10);
            border-radius: 22px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 18px 36px rgba(43, 34, 22, 0.12);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 22px;
            border: 1px solid rgba(18, 32, 39, 0.10);
            overflow: hidden;
            box-shadow: 0 18px 34px rgba(43, 34, 22, 0.10);
        }

        div[data-testid="stImage"] {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(18, 32, 39, 0.10);
            border-radius: 26px;
            padding: 12px;
            box-shadow: 0 18px 34px rgba(43, 34, 22, 0.10);
        }

        div[data-testid="stImage"] img {
            border-radius: 18px;
        }

        div[data-testid="stImage"] [data-testid="stCaptionContainer"] {
            padding-top: 0.35rem;
        }

        .pm-preview-frame {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(18, 32, 39, 0.10);
            border-radius: 26px;
            padding: 12px;
            box-shadow: 0 18px 34px rgba(43, 34, 22, 0.10);
        }

        .pm-preview-shell {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-radius: 18px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.65) 0%, rgba(248, 242, 235, 0.82) 100%);
            border: 1px solid rgba(18, 32, 39, 0.08);
        }

        .pm-preview-shell img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 14px;
            box-shadow: 0 10px 24px rgba(43, 34, 22, 0.10);
        }

        .pm-preview-caption {
            padding-top: 0.45rem;
            color: var(--pm-muted);
            font-size: 0.92rem;
        }

        .pm-hero {
            position: relative;
            overflow: hidden;
            padding: 30px 34px;
            border-radius: 30px;
            color: #fffdf9;
            background:
                radial-gradient(circle at top right, rgba(255,255,255,0.20), transparent 22%),
                radial-gradient(circle at 18% 14%, rgba(255,255,255,0.16), transparent 26%),
                linear-gradient(135deg, #6f7e5d 0%, #9aa08a 52%, #c98a6a 100%);
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 34px 66px rgba(32, 39, 41, 0.22), inset 0 1px 0 rgba(255,255,255,0.12);
            margin-bottom: 18px;
        }

        .pm-hero h1 {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.02;
            letter-spacing: -0.03em;
            color: #fffdf9;
            text-shadow: 0 1px 0 rgba(0, 0, 0, 0.12);
        }

        .pm-hero p {
            margin: 12px 0 0 0;
            max-width: 720px;
            color: rgba(255, 253, 249, 0.96);
            font-size: 1rem;
            text-shadow: 0 1px 0 rgba(0, 0, 0, 0.10);
        }

        .pm-hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .pm-badge {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .pm-panel {
            padding: 20px 22px;
            border-radius: var(--pm-radius-lg);
            background: var(--pm-card);
            border: 1px solid var(--pm-line);
            box-shadow: var(--pm-shadow);
        }

        .pm-panel-title {
            margin: 0 0 6px 0;
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--pm-ink);
        }

        .pm-panel-subtitle {
            margin: 0;
            color: var(--pm-muted);
            font-size: 0.94rem;
        }

        .pm-summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin: 12px 0 4px 0;
        }

        .pm-summary-card {
            padding: 16px 18px;
            border-radius: 22px;
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(18, 32, 39, 0.09);
            box-shadow: 0 18px 32px rgba(53, 41, 22, 0.10);
        }

        .pm-summary-card .label {
            color: var(--pm-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .pm-summary-card .value {
            margin-top: 8px;
            color: var(--pm-ink);
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1;
        }

        .pm-summary-card .meta {
            margin-top: 8px;
            color: var(--pm-muted);
            font-size: 0.9rem;
        }

        .pm-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 8px;
        }

        .pm-chip {
            display: inline-flex;
            align-items: center;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(18, 32, 39, 0.10);
            color: var(--pm-ink);
            font-size: 0.86rem;
            font-weight: 700;
        }

        .pm-reco-card {
            padding: 18px 20px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(255,249,241,0.98) 100%);
            border: 1px solid rgba(18, 32, 39, 0.08);
            box-shadow: 0 20px 38px rgba(49, 38, 25, 0.12);
            margin-bottom: 12px;
        }

        .pm-reco-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
        }

        .pm-rank {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #6f7e5d 0%, #9aa08a 52%, #c98a6a 100%);
            color: #fff7ef;
            font-weight: 800;
            font-size: 1rem;
            flex-shrink: 0;
        }

        .pm-reco-name {
            margin: 0;
            color: var(--pm-ink);
            font-size: 1.14rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .pm-reco-meta {
            color: var(--pm-muted);
            font-size: 0.92rem;
            margin-top: 6px;
        }

        .pm-reco-score {
            color: var(--pm-accent-dark);
            font-weight: 800;
            font-size: 1.05rem;
            white-space: nowrap;
        }

        .pm-reco-reasons {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .pm-soft-pill {
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--pm-accent-soft);
            color: var(--pm-accent-dark);
            font-size: 0.82rem;
            font-weight: 700;
        }

        .pm-section-title {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--pm-ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, help_text: str | None = None):
    st.markdown(
        f"""
        <div class='pm-summary-card'>
            <div class='label'>{title}</div>
            <div class='value'>{value}</div>
            <div class='meta'>{help_text or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_cards(payload: dict, items_df: pd.DataFrame, ocr_df: pd.DataFrame, line_roles_df: pd.DataFrame):
    parser_label = payload.get("parser_module", "unknown").split(".")[-1]
    ocr_backend = payload.get("ocr_backend", "unknown")
    line_role_status = "loaded" if payload.get("line_role_model_loaded") else "not loaded"
    avg_conf = "—"
    if not ocr_df.empty and "ocr_confidence" in ocr_df.columns:
        valid = pd.to_numeric(ocr_df["ocr_confidence"], errors="coerce").dropna()
        if not valid.empty:
            avg_conf = f"{valid.mean():.2f}"

    st.markdown(
        f"""
        <div class='pm-summary-grid'>
            <div class='pm-summary-card'>
                <div class='label'>Parsed dishes</div>
                <div class='value'>{len(items_df)}</div>
                <div class='meta'>Structured menu items</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>OCR engine used</div>
                <div class='value' style='font-size:1.2rem'>{ocr_backend}</div>
                <div class='meta'>{len(ocr_df)} detected text rows</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>OCR confidence</div>
                <div class='value'>{avg_conf}</div>
                <div class='meta'>{len(ocr_df)} detected text rows</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>Active parser</div>
                <div class='value' style='font-size:1.2rem'>{parser_label}</div>
                <div class='meta'>Current API parser module</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>Line-role model</div>
                <div class='value' style='font-size:1.2rem'>{line_role_status}</div>
                <div class='meta'>{len(line_roles_df)} predicted labels</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_chips(payload: dict):
    parser_label = payload.get("parser_module", "unknown").split(".")[-1]
    chips = [
        f"<span class='pm-chip'>OCR: {payload.get('ocr_backend', 'unknown')}</span>",
        f"<span class='pm-chip'>Parser: {parser_label}</span>",
        f"<span class='pm-chip'>Line-role: {'available' if payload.get('line_role_model_loaded') else 'missing'}</span>",
    ]
    st.markdown(f"<div class='pm-chip-row'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_storage_snapshot_tools(api_url: str):
    st.markdown("<p class='pm-section-title'>Storage snapshot</p>", unsafe_allow_html=True)
    st.caption("Normal docker restarts keep PostgreSQL data. The database is wiped only if you run `docker compose down -v`. Use a snapshot as an extra safety backup.")

    snapshot_payload = fetch_api_json(api_url, "/storage/snapshot", method="GET", timeout=120)
    if isinstance(snapshot_payload, dict):
        row_counts = snapshot_payload.get("row_counts", {})
        st.caption(
            f"Current storage: {row_counts.get('menu_sessions', 0)} sessions, "
            f"{row_counts.get('parsed_items', 0)} parsed items, "
            f"{row_counts.get('dish_embeddings', 0)} embeddings."
        )
        snapshot_text = json.dumps(snapshot_payload, ensure_ascii=False, indent=2)
        st.download_button(
            "Download storage snapshot",
            data=snapshot_text,
            file_name="pickmeal_storage_snapshot.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.warning("Could not export a storage snapshot from the API.")

    restore_file = st.file_uploader(
        "Restore storage snapshot",
        type=["json"],
        key="storage_snapshot_restore_uploader",
        help="This replaces the current PostgreSQL storage content with the uploaded snapshot.",
    )
    if restore_file is not None:
        if button_stretch("Restore storage from snapshot", type="secondary"):
            try:
                snapshot = json.loads(restore_file.getvalue().decode("utf-8"))
                response = requests.post(
                    f"{api_url}/storage/snapshot/import",
                    json={"snapshot": snapshot},
                    timeout=180,
                )
                if response.ok:
                    imported = response.json().get("imported_counts", {})
                    st.success(
                        "Snapshot restored: "
                        f"{imported.get('menu_sessions', 0)} sessions, "
                        f"{imported.get('parsed_items', 0)} items, "
                        f"{imported.get('dish_embeddings', 0)} embeddings."
                    )
                else:
                    try:
                        st.error(format_api_error(response.json()))
                    except Exception:
                        st.error(response.text)
            except Exception as exc:
                st.error(str(exc))


def render_panel_header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div class='pm-panel'>
            <div class='pm-panel-title'>{title}</div>
            <div class='pm-panel-subtitle'>{subtitle or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_cards(rec_df: pd.DataFrame):
    if rec_df.empty:
        st.warning("No dishes matched the current filters. Try relaxing allergens, disliked terms, or price limit.")
        return
    if "match_label" in rec_df.columns and rec_df["match_label"].eq("Weak match").all():
        st.warning("Only weak matches were found. Try relaxing the request or preferred section to get stronger recommendations.")

    for _, row in rec_df.iterrows():
        reasons = row.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons_html = "".join(f"<span class='pm-soft-pill'>{reason}</span>" for reason in reasons[:6])
        section = row.get("section") or "No section"
        price = row.get("price_value")
        price_text = f"Price: {price}" if pd.notna(price) else "Price: —"
        calories = row.get("calories_mid")
        calories_text = f"{int(calories)} kcal" if pd.notna(calories) else "kcal: —"
        diet_flags = row.get("diet_flags", [])
        if not isinstance(diet_flags, list):
            diet_flags = []
        diet_html = "".join(f"<span class='pm-soft-pill'>{flag}</span>" for flag in diet_flags[:4])
        match_label = row.get("match_label") or "Match"
        st.markdown(
            f"""
            <div class='pm-reco-card'>
                <div class='pm-reco-top'>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <div class='pm-rank'>{int(row.get('rank', 0))}</div>
                        <div>
                            <p class='pm-reco-name'>{row.get('dish_name', 'Unnamed dish')}</p>
                            <div class='pm-reco-meta'>{section} · {price_text} · {calories_text}</div>
                        </div>
                    </div>
                    <div class='pm-reco-score'>{match_label}</div>
                </div>
                <div class='pm-reco-reasons'>{diet_html}</div>
                <div class='pm-reco-reasons'>{reasons_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_combo_cards(combo_df: pd.DataFrame):
    if combo_df.empty:
        st.info("No feasible dish combinations matched the current budget or calorie limits.")
        return

    for _, row in combo_df.iterrows():
        reasons = row.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons_html = "".join(f"<span class='pm-soft-pill'>{reason}</span>" for reason in reasons[:6])
        dishes = row.get("dish_names", [])
        if not isinstance(dishes, list):
            dishes = []
        sections = row.get("sections", [])
        if not isinstance(sections, list):
            sections = []
        combo_title = " + ".join(dishes[:4]) if dishes else "Dish combination"
        section_text = ", ".join(sorted(set(sections))) if sections else "Mixed sections"
        total_price = row.get("total_price")
        total_price_text = f"Total price: {total_price}" if pd.notna(total_price) else "Total price: —"
        total_calories = row.get("total_calories")
        total_calories_text = f"{int(total_calories)} kcal" if pd.notna(total_calories) else "kcal: —"
        match_label = row.get("match_label") or "Match"

        st.markdown(
            f"""
            <div class='pm-reco-card'>
                <div class='pm-reco-top'>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <div class='pm-rank'>{int(row.get('rank', 0))}</div>
                        <div>
                            <p class='pm-reco-name'>{combo_title}</p>
                            <div class='pm-reco-meta'>{section_text} · {total_price_text} · {total_calories_text}</div>
                        </div>
                    </div>
                    <div class='pm-reco-score'>{match_label}</div>
                </div>
                <div class='pm-reco-reasons'>{reasons_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_llm_cards(llm_df: pd.DataFrame):
    if llm_df.empty:
        st.info("No LLM dish cards are available yet.")
        return

    for _, row in llm_df.iterrows():
        dish_name = row.get("dish_name", "Unnamed dish")
        section = row.get("section") or "No section"
        summary = row.get("llm_summary") or "No summary returned."
        why_it_fits = row.get("llm_why_it_fits") or "No fit note returned."
        caution_note = row.get("llm_caution_note") or "No caution note returned."
        st.markdown(
            f"""
            <div class='pm-reco-card'>
                <div class='pm-reco-top'>
                    <div>
                        <p class='pm-reco-name'>{dish_name}</p>
                        <div class='pm-reco-meta'>{section}</div>
                    </div>
                    <div class='pm-reco-score'>LLM card</div>
                </div>
                <div class='pm-reco-meta' style='margin-top:12px;color:#1f2933;'>{summary}</div>
                <div class='pm-reco-reasons'>
                    <span class='pm-soft-pill'>Why it fits</span>
                    <span>{why_it_fits}</span>
                </div>
                <div class='pm-reco-reasons'>
                    <span class='pm-soft-pill'>Caution</span>
                    <span>{caution_note}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def overlay_ocr_boxes(image: Image.Image, ocr_df: pd.DataFrame) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if ocr_df.empty:
        return canvas
    for _, row in ocr_df.iterrows():
        try:
            x1 = float(row["bbox_x1"])
            y1 = float(row["bbox_y1"])
            x2 = float(row["bbox_x2"])
            y2 = float(row["bbox_y2"])
            draw.rectangle([x1, y1, x2, y2], outline=(37, 99, 235), width=2)
        except Exception:
            continue
    return canvas


def render_dashboard(api_url: str):
    st.subheader("Model dashboard")
    parser_stats = fetch_api_json(api_url, "/stats/parser")
    api_parser_df = pd.DataFrame((parser_stats or {}).get("comparison_rows", []))
    local_parser_df = load_parser_metric_rows()
    parser_df = merge_metric_frames(api_parser_df, local_parser_df, subset=["metrics_dir", "split"])
    active_metrics_dir = normalize_metrics_dir(parser_stats.get("active_metrics_dir")) if isinstance(parser_stats, dict) else ""

    line_role_df = load_line_role_metric_rows()
    ocr_df = load_ocr_metric_rows()

    parser_test_df = pd.DataFrame()
    active_parser_row = {}
    best_parser_row = {}
    best_ocr_row = {}
    best_line_role_row = {}

    if not parser_df.empty:
        parser_test_df = parser_df.loc[parser_df["split"] == "test"].copy()
        parser_test_df = parser_test_df.sort_values(by="item_f1", ascending=False, na_position="last")
        if active_metrics_dir and "metrics_dir" in parser_test_df.columns:
            active_rows = parser_test_df.loc[parser_test_df["metrics_dir"] == active_metrics_dir]
            if not active_rows.empty:
                active_parser_row = active_rows.iloc[0].to_dict()
        if not parser_test_df.empty:
            best_parser_row = parser_test_df.iloc[0].to_dict()

    if not ocr_df.empty:
        ocr_df = ocr_df.sort_values(by="item_f1", ascending=False, na_position="last")
        best_ocr_row = ocr_df.iloc[0].to_dict()

    if not line_role_df.empty:
        line_role_df = line_role_df.sort_values(by="macro_f1", ascending=False, na_position="last")
        best_line_role_row = line_role_df.iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Active parser test F1",
            f"{active_parser_row.get('item_f1', '—')}",
            f"{active_parser_row.get('model', 'Current parser')} on the test split",
        )
    with c2:
        metric_card(
            "Best parser test F1",
            f"{best_parser_row.get('item_f1', '—')}",
            f"{best_parser_row.get('model', 'No parser comparison loaded')}",
        )
    with c3:
        metric_card(
            "Best OCR item F1",
            f"{best_ocr_row.get('item_f1', '—')}",
            f"{best_ocr_row.get('backend', '—')} · {best_ocr_row.get('benchmark_run', '—')}",
        )
    with c4:
        metric_card(
            "Best line-role macro F1",
            f"{best_line_role_row.get('macro_f1', '—')}",
            f"{best_line_role_row.get('model', 'No line-role metrics loaded')}",
        )

    if not parser_df.empty:
        metric_cols = [
            c
            for c in [
                "item_f1",
                "item_precision",
                "item_recall",
                "section_accuracy",
                "price_accuracy",
                "description_exact_match",
                "n_menus",
            ]
            if c in parser_df.columns
        ]
        st.markdown("### Parser comparison")
        parser_table = parser_df.sort_values(by=["split", "item_f1"], ascending=[True, False], na_position="last")
        show_dataframe(parser_table[["model", "split"] + metric_cols], height=280)

        if not parser_test_df.empty:
            chart_df = parser_test_df[["model", "item_f1", "price_accuracy", "section_accuracy"]].copy()
            chart_df = chart_df.set_index("model")
            st.bar_chart(chart_df)

    if not line_role_df.empty:
        st.markdown("### Line-role model comparison")
        cols = [c for c in ["model", "accuracy", "macro_f1", "weighted_f1", "n_train_rows", "n_test_rows"] if c in line_role_df.columns]
        show_dataframe(line_role_df[cols], height=200)

    if not ocr_df.empty:
        st.markdown("### OCR benchmark comparison")
        cols = [
            c
            for c in [
                "benchmark_run",
                "backend",
                "item_f1",
                "item_precision",
                "item_recall",
                "price_accuracy",
                "section_accuracy",
                "avg_ocr_lines",
                "avg_ocr_confidence",
            ]
            if c in ocr_df.columns
        ]
        show_dataframe(ocr_df[cols], height=220)
        ocr_chart_df = ocr_df[["benchmark_run", "backend", "item_f1", "price_accuracy"]].copy()
        ocr_chart_df["label"] = ocr_chart_df["benchmark_run"] + " · " + ocr_chart_df["backend"]
        st.bar_chart(ocr_chart_df.set_index("label")[["item_f1", "price_accuracy"]])

    storage_stats = fetch_api_json(api_url, "/stats/storage")
    if isinstance(storage_stats, dict):
        st.markdown("### Storage and embeddings")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            metric_card("Storage enabled", "yes" if storage_stats.get("enabled") else "no", "Database-backed persistence")
        with s2:
            metric_card("Storage ready", "yes" if storage_stats.get("initialized") else "no", "Postgres-backed persistence initialized")
        with s3:
            metric_card("Stored sessions", str(storage_stats.get("row_counts", {}).get("menu_sessions", "—")), "Persisted parse sessions")
        with s4:
            metric_card("Stored embeddings", str(storage_stats.get("row_counts", {}).get("dish_embeddings", "—")), "Similarity index entries")

        stats_rows = pd.DataFrame(
            [
                {
                    "database_url_present": storage_stats.get("database_url_present"),
                    "vector_backend": storage_stats.get("vector_backend") or "",
                    "embedding_model": storage_stats.get("embedding_model_name"),
                    "embedding_dimensions": storage_stats.get("embedding_dimensions"),
                    "recommendation_runs": storage_stats.get("row_counts", {}).get("recommendation_runs", 0),
                    "source_kind_counts": json.dumps(storage_stats.get("source_kind_counts") or {}, ensure_ascii=False),
                    "last_error": storage_stats.get("last_error") or "",
                }
            ]
        )
        show_dataframe(stats_rows, height=120)

    nutrition_stats = fetch_api_json(api_url, "/stats/nutrition")
    if isinstance(nutrition_stats, dict):
        st.markdown("### Nutrition reference layer")
        n1, n2, n3, n4 = st.columns(4)
        latest_version = nutrition_stats.get("latest_version") or {}
        with n1:
            metric_card("Nutrition enabled", "yes" if nutrition_stats.get("enabled") else "no", "Reference-backed enrichment layer")
        with n2:
            metric_card("Nutrition ready", "yes" if nutrition_stats.get("initialized") else "no", "Versioned seed loaded into PostgreSQL")
        with n3:
            metric_card("Reference rows", str(nutrition_stats.get("row_counts", {}).get("nutrition_reference_items", "—")), "Ingredient-level nutrition records")
        with n4:
            metric_card("Reference version", str(latest_version.get("source_version") or "—"), "Current loaded nutrition reference set")

        nutrition_df = pd.DataFrame(
            [
                {
                    "seed_path": nutrition_stats.get("seed_path") or "",
                    "latest_source": latest_version.get("source_name") or "",
                    "record_count": latest_version.get("record_count") or 0,
                    "source_counts": json.dumps(nutrition_stats.get("source_counts") or {}, ensure_ascii=False),
                    "last_error": nutrition_stats.get("last_error") or "",
                }
            ]
        )
        show_dataframe(nutrition_df, height=120)

    cache_stats = fetch_api_json(api_url, "/stats/cache")
    if isinstance(cache_stats, dict):
        st.markdown("### Redis cache layer")
        c1, c2, c3, c4 = st.columns(4)
        counters = cache_stats.get("local_counters") or {}
        namespace_counts = cache_stats.get("namespace_key_counts") or {}
        with c1:
            metric_card("Cache enabled", "yes" if cache_stats.get("enabled") else "no", "Optional Redis-backed cache layer")
        with c2:
            metric_card("Cache ready", "yes" if cache_stats.get("available") else "no", "Redis ping and client availability")
        with c3:
            metric_card("Cache hits", str(counters.get("hit_count", 0)), "Hits recorded by this service process")
        with c4:
            metric_card("Cached keys", str(sum(int(v) for v in namespace_counts.values())), "Keys under the current namespace")

        cache_df = pd.DataFrame(
            [
                {
                    "configured": cache_stats.get("configured"),
                    "namespace": cache_stats.get("namespace") or "",
                    "recommendation_ttl_seconds": cache_stats.get("recommendation_ttl_seconds"),
                    "llm_ttl_seconds": cache_stats.get("llm_ttl_seconds"),
                    "local_counters": json.dumps(counters, ensure_ascii=False),
                    "namespace_key_counts": json.dumps(namespace_counts, ensure_ascii=False),
                    "server_info": json.dumps(cache_stats.get("server_info") or {}, ensure_ascii=False),
                    "last_error": cache_stats.get("last_error") or "",
                }
            ]
        )
        show_dataframe(cache_df, height=120)

    runtime_stats = fetch_api_json(api_url, "/stats/runtime")
    if isinstance(runtime_stats, dict):
        st.markdown("### Runtime monitoring")
        rt1, rt2, rt3, rt4 = st.columns(4)
        path_rows = runtime_stats.get("path_rows") or []
        stage_rows = runtime_stats.get("stage_rows") or []
        recent_errors = runtime_stats.get("recent_errors") or []
        stage_error_count = int(sum(int(row.get("error_count") or 0) for row in stage_rows))
        with rt1:
            metric_card("Uptime (s)", str(runtime_stats.get("uptime_seconds") or "0"), "Current service process uptime")
        with rt2:
            metric_card("HTTP requests", str(runtime_stats.get("request_count") or 0), "Requests seen by the gateway")
        with rt3:
            metric_card("Tracked routes", str(len(path_rows)), "Per-route latency and status counters")
        with rt4:
            metric_card("Stage errors", str(stage_error_count), "Failures captured across OCR, parser, storage, and recommendation steps")

        runtime_df = pd.DataFrame(path_rows)
        if not runtime_df.empty:
            cols = [c for c in ["method", "path", "count", "error_count", "last_status", "avg_duration_ms", "max_duration_ms", "last_seen_at"] if c in runtime_df.columns]
            st.markdown("#### Route-level request stats")
            show_dataframe(runtime_df[cols], height=220)

        stage_df = pd.DataFrame(stage_rows)
        if not stage_df.empty:
            cols = [c for c in ["component", "operation", "count", "ok_count", "error_count", "avg_duration_ms", "max_duration_ms", "last_duration_ms"] if c in stage_df.columns]
            st.markdown("#### Stage-level pipeline stats")
            show_dataframe(stage_df[cols], height=220)

        if recent_errors:
            error_df = pd.DataFrame(recent_errors)
            st.markdown("#### Recent runtime errors")
            show_dataframe(error_df, height=180)

    llm_stats = fetch_api_json(api_url, "/stats/llm")
    if isinstance(llm_stats, dict):
        st.markdown("### LLM enrichment")
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            metric_card("LLM enabled", "yes" if llm_stats.get("enabled") else "no", "Feature flag status")
        with l2:
            metric_card("LLM configured", "yes" if llm_stats.get("configured") else "no", "Base URL, key, and model")
        with l3:
            metric_card("LLM model", str(llm_stats.get("model") or "—"), "Configured external model")
        with l4:
            metric_card("Max items per call", str(llm_stats.get("max_items_per_request") or "—"), "Scope-limited enrichment")
        llm_df = pd.DataFrame(
            [
                {
                    "base_url_present": llm_stats.get("base_url_present"),
                    "api_key_present": llm_stats.get("api_key_present"),
                    "timeout_seconds": llm_stats.get("timeout_seconds"),
                    "last_error": llm_stats.get("last_error") or "",
                }
            ]
        )
        show_dataframe(llm_df, height=120)

    rag_stats = fetch_api_json(api_url, "/stats/rag")
    if isinstance(rag_stats, dict):
        st.markdown("### RAG knowledge base")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            metric_card("RAG enabled", "yes" if rag_stats.get("enabled") else "no", "Grounded retrieval layer")
        with r2:
            metric_card("RAG ready", "yes" if rag_stats.get("initialized") else "no", "Knowledge base seeded in PostgreSQL")
        with r3:
            metric_card("Knowledge docs", str(rag_stats.get("document_count") or 0), "Retrieved evidence corpus")
        with r4:
            metric_card("RAG top-k", str(rag_stats.get("top_k") or "—"), "Evidence chunks per item")
        rag_df = pd.DataFrame(
            [
                {
                    "vector_backend": rag_stats.get("vector_backend") or "",
                    "dataset_path": rag_stats.get("dataset_path") or "",
                    "source_type_counts": json.dumps(rag_stats.get("source_type_counts") or {}, ensure_ascii=False),
                    "last_error": rag_stats.get("last_error") or "",
                }
            ]
        )
        show_dataframe(rag_df, height=120)


def render_dataset_tab():
    st.subheader("Dataset overview")

    summary = read_json_if_exists(PROJECT_ROOT / "data" / "interim" / "dataset_summary.json")
    if summary:
        st.json(summary)
    else:
        parser_valid = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_cascade_v1_expanded" / "parser_metrics_valid.json") or {}
        parser_test = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_cascade_v1_expanded" / "parser_metrics_test.json") or {}
        line_role = read_json_if_exists(PROJECT_ROOT / "reports" / "line_role_expanded_sgd_v1" / "line_role_metrics.json") or {}
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Parser valid menus", f"{parser_valid.get('n_menus', '—')}", "Expanded validation split")
        with c2:
            metric_card("Parser test menus", f"{parser_test.get('n_menus', '—')}", "Expanded held-out split")
        with c3:
            metric_card("Line-role test rows", f"{line_role.get('n_test_rows', '—')}", "Expanded OCR line evaluation set")
        st.caption("Detailed dataset summary is not tracked in Git, so this tab falls back to committed benchmark artifacts.")

    rf_manifest = read_csv_if_exists(PROJECT_ROOT / "data" / "interim" / "roboflow_manifest.csv")
    kg_manifest = read_csv_if_exists(PROJECT_ROOT / "data" / "interim" / "kaggle_manifest.csv")

    if not rf_manifest.empty:
        st.markdown("### Roboflow split sizes")
        split_df = rf_manifest.groupby("split", as_index=False).size().rename(columns={"size": "n_images"})
        show_dataframe(split_df, height=180)
        st.bar_chart(split_df.set_index("split")[["n_images"]])

    if not kg_manifest.empty:
        st.markdown("### Kaggle corpus")
        kaggle_df = pd.DataFrame(
            [
                {
                    "n_images_total": int(len(kg_manifest)),
                    "n_valid_images": int((kg_manifest["ok"] == 1).sum()) if "ok" in kg_manifest.columns else int(len(kg_manifest)),
                    "n_tiny_images": int(kg_manifest["is_tiny"].sum()) if "is_tiny" in kg_manifest.columns else 0,
                }
            ]
        )
        show_dataframe(kaggle_df)


def render_how_it_works():
    st.markdown(
        """
        **How the pipeline works**
        1. The image is loaded and optionally rotated.
        2. OCR extracts text lines with coordinates.
        3. The parser builds menu items from names, descriptions, prices, and sections.
        4. The line-role model labels OCR rows as item, price, section, description, or noise.
        5. A nutrition layer adds ingredient hints, allergen signals, diet flags, and calorie ranges.
        6. The recommendation layer applies hard filters, ranks relevant dishes, and can build feasible dish combinations under budget and calorie limits.
        7. An optional LLM layer can generate user-facing dish cards and short explanation text.
        """
    )


def render_similar_dishes_lookup(api_url: str):
    st.markdown("<p class='pm-section-title'>Similar parsed dishes in local storage</p>", unsafe_allow_html=True)
    st.caption("Search across dishes persisted from previous parsing sessions and the gold-seeded menu corpus stored in PostgreSQL with pgvector.")
    with st.form("similar_dishes_form"):
        query_text = st.text_input("Similarity query", value="margherita pizza", key="similarity_query_input")
        top_k = st.slider("Top similar dishes", min_value=3, max_value=10, value=5, key="similarity_top_k_slider")
        find_clicked = st.form_submit_button("Find similar dishes")

    if find_clicked:
        similar_payload = fetch_api_json(
            api_url,
            "/storage/similar",
            method="POST",
            json_payload={"query_text": query_text, "top_k": top_k},
            timeout=60,
        )
        if isinstance(similar_payload, dict):
            similar_df = pd.DataFrame(similar_payload.get("rows", []))
            if similar_df.empty:
                st.info("No stored dishes matched yet. Parse at least one menu first.")
            else:
                show_dataframe(similar_df, height=220)
        else:
            st.warning("Could not load similar dishes from the storage API.")


st.set_page_config(page_title="PickMeal AI", layout="wide")
init_state()
inject_styles()

st.markdown(
    """
    <div class='pm-hero'>
        <h1>PickMeal AI</h1>
        <p>Scan restaurant menus, inspect OCR quality, parse dishes into structure, and surface recommendations.</p>
        <div class='pm-hero-badges'>
            <span class='pm-badge'>OCR + parser pipeline</span>
            <span class='pm-badge'>Rotatable image input</span>
            <span class='pm-badge'>Parser metrics and diagnostics</span>
            <span class='pm-badge'>Nutrition-aware recommendations</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    st.text_input("FastAPI URL", key="api_url_input")
    api_url = normalize_api_url(st.session_state.get("api_url_input"))
    st.caption("Run the API first, then open this app.")
    show_developer_tools = st.checkbox("Developer mode", value=False, key="developer_mode_toggle")

home_tab, diagnostics_tab, metrics_tab, data_tab = st.tabs(
    ["Live demo", "Parsing diagnostics", "Model metrics", "Dataset"]
)

with home_tab:
    control_col, preview_col = st.columns([0.88, 1.12])
    with control_col:
        st.markdown("<p class='pm-section-title'>Prepare the menu image</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a menu image", type=["jpg", "jpeg", "png", "webp"], key="menu_image_uploader")
        langs = st.text_input("OCR languages", value="en", key="ocr_langs_input")
        ocr_backend_default_index = OCR_BACKEND_OPTIONS.index(DEFAULT_OCR_BACKEND) if DEFAULT_OCR_BACKEND in OCR_BACKEND_OPTIONS else 0
        ocr_backend = st.selectbox(
            "OCR engine",
            options=OCR_BACKEND_OPTIONS,
            index=ocr_backend_default_index,
            format_func=lambda option: OCR_BACKEND_LABELS.get(option, option),
            key="ocr_backend_select",
        )
        st.caption(f"Selected OCR engine: {OCR_BACKEND_LABELS.get(ocr_backend, ocr_backend)}")
        rotation_deg = st.selectbox("Rotate image", options=[0, 90, 180, 270], format_func=lambda x: f"{x}°", index=0, key="rotation_deg_select")
        st.caption("Use rotation before parsing so the menu is upright for OCR and the parser.")

    upload_signature = None
    prepared_upload_bytes = None
    prepared_upload_mime = None
    rotated_preview = None

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        preview_image = load_preview_image(image_bytes)
        rotated_preview = rotate_image(preview_image, rotation_deg)
        with preview_col:
            st.markdown("<p class='pm-section-title'>Preview</p>", unsafe_allow_html=True)
            render_preview_frame(rotated_preview, caption=f"Menu image ({rotation_deg}°)")
        prepared_upload_bytes, prepared_upload_mime = image_to_upload_bytes(rotated_preview, uploaded.name)
        upload_signature = (uploaded.name, len(image_bytes), langs, ocr_backend, rotation_deg)

        if st.session_state["parsed_signature"] is not None and st.session_state["parsed_signature"] != upload_signature:
            st.session_state["parsed_payload"] = None
            st.session_state["parse_error"] = None
            st.session_state["recommend_payload"] = None
            st.session_state["recommend_error"] = None
            st.session_state["llm_payload"] = None
            st.session_state["llm_error"] = None
    else:
        st.session_state["parsed_payload"] = None
        st.session_state["parse_error"] = None
        st.session_state["recommend_payload"] = None
        st.session_state["recommend_error"] = None
        st.session_state["llm_payload"] = None
        st.session_state["llm_error"] = None
        st.session_state["parsed_signature"] = None

    with control_col:
        parse_clicked = button_stretch("Parse menu", type="primary")

    if parse_clicked:
        if uploaded is None or prepared_upload_bytes is None:
            st.warning("Please upload a menu image before parsing.")
        else:
            files = {"file": (uploaded.name, prepared_upload_bytes, prepared_upload_mime or "image/png")}
            data = {"langs": langs, "backend": ocr_backend}
            try:
                response = requests.post(f"{api_url}/parse/image", files=files, data=data, timeout=300)
                if response.ok:
                    st.session_state["parsed_payload"] = response.json()
                    st.session_state["parse_error"] = None
                    st.session_state["recommend_payload"] = None
                    st.session_state["recommend_error"] = None
                    st.session_state["llm_payload"] = None
                    st.session_state["llm_error"] = None
                    st.session_state["parsed_signature"] = upload_signature
                else:
                    try:
                        st.session_state["parse_error"] = response.json()
                    except Exception:
                        st.session_state["parse_error"] = response.text
                    st.session_state["parsed_payload"] = None
            except Exception as exc:
                st.session_state["parse_error"] = str(exc)
                st.session_state["parsed_payload"] = None

    if st.session_state["parse_error"] is not None:
        st.error(format_api_error(st.session_state["parse_error"]))

    payload = st.session_state.get("parsed_payload")
    if payload:
        items_df = pd.DataFrame(payload.get("items", []))
        ocr_df = pd.DataFrame(payload.get("ocr_lines", []))
        line_roles_df = pd.DataFrame(payload.get("line_roles", []))

        st.success(f"Parsed {len(items_df)} items from {len(ocr_df)} OCR lines")
        if payload.get("session_id"):
            st.caption(f"Stored session: {payload['session_id']}")
        render_status_chips(payload)
        render_summary_cards(payload, items_df, ocr_df, line_roles_df)

        render_how_it_works()

        main_col, side_col = st.columns([1.4, 1.0])
        with main_col:
            st.markdown("<p class='pm-section-title'>Parsed dishes</p>", unsafe_allow_html=True)
            preview_cols = [
                c
                for c in [
                    "dish_name",
                    "section",
                    "price_value",
                    "calories_mid",
                    "ingredient_hints",
                    "explicit_allergens",
                    "diet_flags",
                    "parser_confidence",
                    "nutrition_confidence",
                ]
                if c in items_df.columns
            ]
            show_dataframe(items_df[preview_cols] if preview_cols else items_df, height=360)
        with side_col:
            st.markdown("<p class='pm-section-title'>OCR lines</p>", unsafe_allow_html=True)
            show_dataframe(ocr_df[["line_order", "text", "ocr_confidence"]] if not ocr_df.empty else ocr_df, height=360)

        st.markdown("<p class='pm-section-title'>RAG + LLM dish cards</p>", unsafe_allow_html=True)
        st.caption("Retrieve grounded evidence from the PickMeal knowledge base, then generate concise user-facing dish cards.")
        with st.form("llm_enrichment_form"):
            llm_col1, llm_col2 = st.columns([1.5, 1.0])
            with llm_col1:
                llm_context = st.text_input(
                    "User preference note",
                    value="Prefer lighter dishes with clear allergen notes.",
                    key="llm_context_input",
                )
            with llm_col2:
                llm_top_k = st.slider("How many items to enrich", min_value=1, max_value=6, value=4, key="llm_top_k_slider")
            llm_clicked = st.form_submit_button("Generate LLM dish cards")

        if llm_clicked:
            llm_request = {
                "items": payload["items"],
                "user_context": llm_context,
                "top_k": llm_top_k,
            }
            try:
                response = requests.post(f"{api_url}/llm/enrich-items", json=llm_request, timeout=180)
                if response.ok:
                    st.session_state["llm_payload"] = response.json()
                    st.session_state["llm_error"] = None
                else:
                    try:
                        st.session_state["llm_error"] = response.json()
                    except Exception:
                        st.session_state["llm_error"] = response.text
                    st.session_state["llm_payload"] = None
            except Exception as exc:
                st.session_state["llm_error"] = str(exc)
                st.session_state["llm_payload"] = None

        llm_error = st.session_state.get("llm_error")
        if llm_error:
            st.warning(format_api_error(llm_error))

        llm_payload = st.session_state.get("llm_payload")
        if llm_payload:
            st.info(f"LLM provider: {llm_payload.get('provider_label')} · model: {llm_payload.get('model')}")
            llm_df = pd.DataFrame(llm_payload.get("items", []))
            render_llm_cards(llm_df)
            retrieval_df = pd.DataFrame(llm_payload.get("retrieval_rows", []))
            if not retrieval_df.empty:
                with st.expander("Retrieved RAG evidence"):
                    show_dataframe(retrieval_df, height=260)
            with st.expander("LLM item table"):
                show_dataframe(llm_df, height=240)

        st.markdown("<p class='pm-section-title'>Dish recommendation</p>", unsafe_allow_html=True)
        st.caption("Use user preferences to rank parsed dishes. The recommendation engine is selected automatically unless you override it.")
        with st.form("recommend_form"):
            pref_col1, pref_col2 = st.columns(2)
            with pref_col1:
                craving_text = st.text_input("What do you want right now?", value="creamy pasta with chicken", key="craving_text_input")
                liked_terms = st.text_input("Liked terms (comma-separated)", value="chicken, cheese", key="liked_terms_input")
                disliked_terms = st.text_input("Disliked terms (comma-separated)", value="fish", key="disliked_terms_input")
            with pref_col2:
                excluded_allergens = st.text_input("Excluded allergens (comma-separated)", value="peanut", key="excluded_allergens_input")
                required_diet_flags = st.text_input("Required diet flags (comma-separated)", value="", key="required_diet_flags_input")
                preferred_sections = st.text_input("Preferred menu sections (comma-separated)", value="pasta", key="preferred_sections_input")
                max_price = st.number_input("Max dish price", min_value=0.0, value=500.0, key="max_price_input")
                max_calories = st.number_input("Max calories", min_value=0.0, value=700.0, key="max_calories_input")
                engine = st.selectbox(
                    "Recommendation engine",
                    ["auto", "tfidf", "sentence_transformer", "catboost_reranker"],
                    index=0,
                    key="recommend_engine_select",
                )
            combo_col1, combo_col2 = st.columns(2)
            with combo_col1:
                combo_budget = st.number_input("Combination budget", min_value=0.0, value=0.0, key="combo_budget_input")
                combo_min_items = st.selectbox("Min dishes in combo", options=[2, 3], index=0, key="combo_min_items_select")
            with combo_col2:
                combo_max_calories = st.number_input("Combination max calories", min_value=0.0, value=0.0, key="combo_max_calories_input")
                combo_max_items = st.selectbox("Max dishes in combo", options=[2, 3], index=1, key="combo_max_items_select")
            recommend_clicked = st.form_submit_button("Recommend dishes")

        if recommend_clicked:
            rec_request = {
                "session_id": payload.get("session_id"),
                "items": payload["items"],
                "craving_text": craving_text,
                "liked_terms": [x.strip() for x in liked_terms.split(",") if x.strip()],
                "disliked_terms": [x.strip() for x in disliked_terms.split(",") if x.strip()],
                "excluded_allergens": [x.strip() for x in excluded_allergens.split(",") if x.strip()],
                "required_diet_flags": [x.strip() for x in required_diet_flags.split(",") if x.strip()],
                "preferred_sections": [x.strip() for x in preferred_sections.split(",") if x.strip()],
                "max_price": max_price,
                "max_calories": max_calories,
                "combo_budget": combo_budget if combo_budget > 0 else None,
                "combo_max_calories": combo_max_calories if combo_max_calories > 0 else None,
                "combo_min_items": combo_min_items,
                "combo_max_items": combo_max_items,
                "top_k": 5,
                "engine": engine,
            }
            try:
                response = requests.post(f"{api_url}/recommend", json=rec_request, timeout=180)
                if response.ok:
                    st.session_state["recommend_payload"] = response.json()
                    st.session_state["recommend_error"] = None
                else:
                    try:
                        st.session_state["recommend_error"] = response.json()
                    except Exception:
                        st.session_state["recommend_error"] = response.text
                    st.session_state["recommend_payload"] = None
            except Exception as exc:
                st.session_state["recommend_error"] = str(exc)
                st.session_state["recommend_payload"] = None

        recommend_payload = st.session_state.get("recommend_payload")
        recommend_error = st.session_state.get("recommend_error")
        if recommend_error:
            st.error(format_api_error(recommend_error))
        if recommend_payload:
            st.info(f"Engine used: {recommend_payload.get('engine_used')}")
            if recommend_payload.get("recommendation_id"):
                st.caption(f"Stored recommendation run: {recommend_payload['recommendation_id']}")
            rec_df = pd.DataFrame(recommend_payload.get("rows", []))
            render_recommendation_cards(rec_df)
            combo_df = pd.DataFrame(recommend_payload.get("combo_rows", []))
            st.markdown("<p class='pm-section-title'>Dish combinations</p>", unsafe_allow_html=True)
            render_combo_cards(combo_df)
            with st.expander("Recommendation table"):
                show_dataframe(rec_df, height=260)
            if not combo_df.empty:
                with st.expander("Combination table"):
                    show_dataframe(combo_df, height=260)

        render_similar_dishes_lookup(api_url)
        render_storage_snapshot_tools(api_url)

        if show_developer_tools:
            st.markdown("### Raw response")
            st.json(payload)

with diagnostics_tab:
    payload = st.session_state.get("parsed_payload")
    if not payload:
        st.info("Parse an image first to inspect OCR diagnostics.")
    else:
        ocr_df = pd.DataFrame(payload.get("ocr_lines", []))
        items_df = pd.DataFrame(payload.get("items", []))
        line_roles_df = pd.DataFrame(payload.get("line_roles", []))

        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            preview_image = load_preview_image(image_bytes)
            rotated_preview = rotate_image(preview_image, rotation_deg)
            overlay = overlay_ocr_boxes(rotated_preview, ocr_df)
            st.markdown("<p class='pm-section-title'>OCR overlay</p>", unsafe_allow_html=True)
            show_image(overlay, caption="OCR boxes")

        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("<p class='pm-section-title'>OCR table</p>", unsafe_allow_html=True)
            show_dataframe(ocr_df, height=320)
        with diag_col2:
            st.markdown("<p class='pm-section-title'>Line-role table</p>", unsafe_allow_html=True)
            if line_roles_df.empty:
                st.info("Line-role model is not available in the current environment.")
            else:
                show_dataframe(line_roles_df, height=320)

        st.markdown("<p class='pm-section-title'>Parsed items table</p>", unsafe_allow_html=True)
        show_dataframe(items_df, height=280)

with metrics_tab:
    render_dashboard(api_url)

with data_tab:
    render_dataset_tab()

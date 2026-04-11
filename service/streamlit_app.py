from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

API_URL = os.getenv("PICKMEAL_API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def init_state():
    defaults = {
        "parsed_payload": None,
        "parse_error": None,
        "recommend_payload": None,
        "recommend_error": None,
        "parsed_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
                <div class='label'>OCR lines</div>
                <div class='value'>{len(ocr_df)}</div>
                <div class='meta'>Detected text rows</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>Active parser</div>
                <div class='value' style='font-size:1.2rem'>{parser_label}</div>
                <div class='meta'>Current API parser module</div>
            </div>
            <div class='pm-summary-card'>
                <div class='label'>Line-role model</div>
                <div class='value' style='font-size:1.2rem'>{line_role_status}</div>
                <div class='meta'>{len(line_roles_df)} labels, avg OCR conf {avg_conf}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_chips(payload: dict):
    parser_label = payload.get("parser_module", "unknown").split(".")[-1]
    chips = [
        f"<span class='pm-chip'>Parser: {parser_label}</span>",
        f"<span class='pm-chip'>Line-role: {'available' if payload.get('line_role_model_loaded') else 'missing'}</span>",
    ]
    st.markdown(f"<div class='pm-chip-row'>{''.join(chips)}</div>", unsafe_allow_html=True)


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

    for _, row in rec_df.iterrows():
        reasons = row.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons_html = "".join(f"<span class='pm-soft-pill'>{reason}</span>" for reason in reasons[:6])
        section = row.get("section") or "No section"
        price = row.get("price_value")
        price_text = f"Price: {price}" if pd.notna(price) else "Price: —"
        st.markdown(
            f"""
            <div class='pm-reco-card'>
                <div class='pm-reco-top'>
                    <div style='display:flex;gap:14px;align-items:flex-start;'>
                        <div class='pm-rank'>{int(row.get('rank', 0))}</div>
                        <div>
                            <p class='pm-reco-name'>{row.get('dish_name', 'Unnamed dish')}</p>
                            <div class='pm-reco-meta'>{section} · {price_text}</div>
                        </div>
                    </div>
                    <div class='pm-reco-score'>score {row.get('score', '—')}</div>
                </div>
                <div class='pm-reco-reasons'>{reasons_html}</div>
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


def render_dashboard():
    st.subheader("Model dashboard")

    parser_v2_valid = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_baseline_v2" / "parser_metrics_valid.json") or {}
    parser_v2_test = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_baseline_v2" / "parser_metrics_test.json") or {}
    parser_v3_layout_valid = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_line_role_v3_layout" / "parser_metrics_valid.json") or {}
    parser_v3_layout_test = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_line_role_v3_layout" / "parser_metrics_test.json") or {}
    parser_hybrid_valid = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_hybrid_merge_v31" / "parser_metrics_valid.json") or {}
    parser_hybrid_test = read_json_if_exists(PROJECT_ROOT / "reports" / "parser_hybrid_merge_v31" / "parser_metrics_test.json") or {}
    line_role = read_json_if_exists(PROJECT_ROOT / "reports" / "line_role_baseline" / "line_role_metrics.json") or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Active parser test F1", f"{parser_v2_test.get('item_f1', '—')}", "Current stable service parser")
    with c2:
        metric_card("Hybrid parser test F1", f"{parser_hybrid_test.get('item_f1', '—')}", "Best offline report in repo")
    with c3:
        metric_card("Line-role macro F1", f"{line_role.get('macro_f1', '—')}", "OCR line classifier")
    with c4:
        metric_card("Parser gold menus", "39", "Current labeled parser set")

    rows = []
    if parser_v2_valid:
        rows.append({"model": "Parser v2", "split": "valid", **parser_v2_valid})
    if parser_v2_test:
        rows.append({"model": "Parser v2", "split": "test", **parser_v2_test})
    if parser_v3_layout_valid:
        rows.append({"model": "Parser v3 layout", "split": "valid", **parser_v3_layout_valid})
    if parser_v3_layout_test:
        rows.append({"model": "Parser v3 layout", "split": "test", **parser_v3_layout_test})
    if parser_hybrid_valid:
        rows.append({"model": "Parser hybrid v31", "split": "valid", **parser_hybrid_valid})
    if parser_hybrid_test:
        rows.append({"model": "Parser hybrid v31", "split": "test", **parser_hybrid_test})

    if rows:
        comp_df = pd.DataFrame(rows)
        metric_cols = [
            c
            for c in [
                "item_f1",
                "item_precision",
                "item_recall",
                "section_accuracy",
                "price_accuracy",
                "description_exact_match",
            ]
            if c in comp_df.columns
        ]
        st.markdown("### Parser comparison")
        show_dataframe(comp_df[["model", "split"] + metric_cols], height=260)
        for metric in metric_cols:
            plot_df = comp_df[["model", "split", metric]].copy()
            plot_df["label"] = plot_df["model"] + " · " + plot_df["split"]
            st.markdown(f"**{metric}**")
            st.bar_chart(plot_df.set_index("label")[[metric]])

    cm_path = PROJECT_ROOT / "reports" / "line_role_baseline" / "line_role_confusion_matrix.csv"
    cm_df = read_csv_if_exists(cm_path)
    if not cm_df.empty:
        st.markdown("### Line-role confusion matrix")
        show_dataframe(cm_df, height=260)


def render_dataset_tab():
    st.subheader("Dataset overview")

    summary = read_json_if_exists(PROJECT_ROOT / "data" / "interim" / "dataset_summary.json")
    if summary:
        st.json(summary)
    else:
        st.info("Dataset summary is not available yet.")

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
        5. The recommendation layer ranks only relevant dishes and can return no result when the page does not match the request.
        """
    )


st.set_page_config(page_title="PickMeal AI", layout="wide")
init_state()
inject_styles()

st.markdown(
    """
    <div class='pm-hero'>
        <h1>PickMeal AI</h1>
        <p>Scan restaurant menus, inspect OCR quality, parse dishes into structure, and surface recommendations in a product-style demo instead of a raw technical dashboard.</p>
        <div class='pm-hero-badges'>
            <span class='pm-badge'>OCR + parser pipeline</span>
            <span class='pm-badge'>Rotatable image input</span>
            <span class='pm-badge'>Parser metrics and diagnostics</span>
            <span class='pm-badge'>Recommendation baseline</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", value=API_URL)
    st.caption("Run the API first, then open this app.")
    show_developer_tools = st.checkbox("Developer mode", value=False)

home_tab, diagnostics_tab, metrics_tab, data_tab = st.tabs(
    ["Live demo", "Parsing diagnostics", "Model metrics", "Dataset"]
)

with home_tab:
    control_col, preview_col = st.columns([0.88, 1.12])
    with control_col:
        st.markdown("<p class='pm-section-title'>Prepare the menu image</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a menu image", type=["jpg", "jpeg", "png", "webp"])
        langs = st.text_input("OCR languages", value="en")
        rotation_deg = st.selectbox("Rotate image", options=[0, 90, 180, 270], format_func=lambda x: f"{x}°", index=0)
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
            show_image(rotated_preview, caption=f"Menu image ({rotation_deg}°)")
        prepared_upload_bytes, prepared_upload_mime = image_to_upload_bytes(rotated_preview, uploaded.name)
        upload_signature = (uploaded.name, len(image_bytes), langs, rotation_deg)

        if st.session_state["parsed_signature"] is not None and st.session_state["parsed_signature"] != upload_signature:
            st.session_state["parsed_payload"] = None
            st.session_state["parse_error"] = None
            st.session_state["recommend_payload"] = None
            st.session_state["recommend_error"] = None
    else:
        st.session_state["parsed_payload"] = None
        st.session_state["parse_error"] = None
        st.session_state["recommend_payload"] = None
        st.session_state["recommend_error"] = None
        st.session_state["parsed_signature"] = None

    with control_col:
        parse_clicked = button_stretch("Parse menu", type="primary")

    if parse_clicked:
        if uploaded is None or prepared_upload_bytes is None:
            st.warning("Please upload an image first.")
        else:
            files = {"file": (uploaded.name, prepared_upload_bytes, prepared_upload_mime or "image/png")}
            data = {"langs": langs}
            try:
                response = requests.post(f"{api_url}/parse/image", files=files, data=data, timeout=300)
                if response.ok:
                    st.session_state["parsed_payload"] = response.json()
                    st.session_state["parse_error"] = None
                    st.session_state["recommend_payload"] = None
                    st.session_state["recommend_error"] = None
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
        st.error(st.session_state["parse_error"])

    payload = st.session_state.get("parsed_payload")
    if payload:
        items_df = pd.DataFrame(payload.get("items", []))
        ocr_df = pd.DataFrame(payload.get("ocr_lines", []))
        line_roles_df = pd.DataFrame(payload.get("line_roles", []))

        st.success(f"Parsed {len(items_df)} items from {len(ocr_df)} OCR lines")
        render_status_chips(payload)
        render_summary_cards(payload, items_df, ocr_df, line_roles_df)

        render_how_it_works()

        main_col, side_col = st.columns([1.4, 1.0])
        with main_col:
            st.markdown("<p class='pm-section-title'>Parsed dishes</p>", unsafe_allow_html=True)
            show_dataframe(items_df, height=360)
        with side_col:
            st.markdown("<p class='pm-section-title'>OCR lines</p>", unsafe_allow_html=True)
            show_dataframe(ocr_df[["line_order", "text", "ocr_confidence"]] if not ocr_df.empty else ocr_df, height=360)

        st.markdown("<p class='pm-section-title'>Dish recommendation</p>", unsafe_allow_html=True)
        st.caption("Use user preferences to rank parsed dishes. The recommendation engine is selected automatically unless you override it.")
        with st.form("recommend_form"):
            pref_col1, pref_col2 = st.columns(2)
            with pref_col1:
                craving_text = st.text_input("What do you want right now?", value="creamy pasta with chicken")
                liked_terms = st.text_input("Liked terms (comma-separated)", value="chicken, cheese")
                disliked_terms = st.text_input("Disliked terms (comma-separated)", value="fish")
            with pref_col2:
                excluded_allergens = st.text_input("Excluded allergens (comma-separated)", value="peanut")
                preferred_sections = st.text_input("Preferred menu sections (comma-separated)", value="pasta")
                max_price = st.number_input("Max price", min_value=0.0, value=500.0)
                engine = st.selectbox("Recommendation engine", ["auto", "tfidf", "sentence_transformer"], index=0)
            recommend_clicked = st.form_submit_button("Recommend dishes")

        if recommend_clicked:
            rec_request = {
                "items": payload["items"],
                "craving_text": craving_text,
                "liked_terms": [x.strip() for x in liked_terms.split(",") if x.strip()],
                "disliked_terms": [x.strip() for x in disliked_terms.split(",") if x.strip()],
                "excluded_allergens": [x.strip() for x in excluded_allergens.split(",") if x.strip()],
                "preferred_sections": [x.strip() for x in preferred_sections.split(",") if x.strip()],
                "max_price": max_price,
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
            st.error(recommend_error)
        if recommend_payload:
            st.info(f"Engine used: {recommend_payload.get('engine_used')}")
            rec_df = pd.DataFrame(recommend_payload.get("rows", []))
            render_recommendation_cards(rec_df)
            with st.expander("Recommendation table"):
                show_dataframe(rec_df, height=260)

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
    render_dashboard()

with data_tab:
    render_dataset_tab()

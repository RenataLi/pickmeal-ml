from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("PICKMEAL_API_URL", "http://localhost:8000")


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


def show_dataframe(df: pd.DataFrame):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True)
        except TypeError:
            st.dataframe(df)


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


st.set_page_config(page_title="PickMeal demo", layout="wide")
init_state()

st.title("PickMeal — menu parsing demo")
st.caption("Upload a menu image, inspect OCR, parse structured items, and run a recommendation baseline.")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("FastAPI URL", value=API_URL)
    st.markdown("Run API first, then use this UI.")

uploaded = st.file_uploader("Upload menu image", type=["jpg", "jpeg", "png", "webp"])
langs = st.text_input("OCR languages", value="en")

upload_signature = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    upload_signature = (uploaded.name, len(image_bytes), langs)
    image = Image.open(uploaded)
    show_image(image, caption="Uploaded menu")

    # Clear stale results when a new file or OCR language selection is used.
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

parse_clicked = button_stretch("Parse image", type="primary")

if parse_clicked:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        files = {"file": (uploaded.name, image_bytes, uploaded.type or "image/png")}
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
    st.success(f"Parsed {payload['n_items']} items from {payload['n_lines']} OCR lines")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("OCR lines")
        show_dataframe(pd.DataFrame(payload["ocr_lines"]))
    with col2:
        st.subheader("Parsed items")
        items_df = pd.DataFrame(payload["items"])
        show_dataframe(items_df)

    st.subheader("Recommendation baseline")
    with st.form("recommend_form"):
        craving_text = st.text_input("What do you want right now?", value="creamy pasta with chicken")
        liked_terms = st.text_input("Liked terms (comma-separated)", value="chicken, cheese")
        disliked_terms = st.text_input("Disliked terms (comma-separated)", value="fish")
        excluded_allergens = st.text_input("Excluded allergens (comma-separated)", value="peanut")
        preferred_sections = st.text_input("Preferred sections (comma-separated)", value="pasta")
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
            rec_response = requests.post(f"{api_url}/recommend", json=rec_request, timeout=300)
            if rec_response.ok:
                st.session_state["recommend_payload"] = rec_response.json()
                st.session_state["recommend_error"] = None
            else:
                try:
                    st.session_state["recommend_error"] = rec_response.json()
                except Exception:
                    st.session_state["recommend_error"] = rec_response.text
                st.session_state["recommend_payload"] = None
        except Exception as exc:
            st.session_state["recommend_error"] = str(exc)
            st.session_state["recommend_payload"] = None

    if st.session_state["recommend_error"] is not None:
        st.error(st.session_state["recommend_error"])

    rec_payload = st.session_state.get("recommend_payload")
    if rec_payload is not None:
        st.info(f"Engine used: {rec_payload['engine_used']}")
        rows_df = pd.DataFrame(rec_payload.get("rows", []))
        if rows_df.empty:
            st.warning("No dishes matched the current filters. Try relaxing allergens, disliked terms, or price limit.")
        else:
            show_dataframe(rows_df)

st.divider()
st.subheader("Service stats")
col1, col2 = st.columns(2)
with col1:
    if st.button("Load dataset stats"):
        response = requests.get(f"{api_url}/stats/dataset", timeout=60)
        if response.ok:
            st.json(response.json())
        else:
            st.error(response.text)
with col2:
    if st.button("Load parser stats"):
        response = requests.get(f"{api_url}/stats/parser", timeout=60)
        if response.ok:
            payload = response.json()
            st.json(payload["metrics"])
            if payload["comparison_rows"]:
                show_dataframe(pd.DataFrame(payload["comparison_rows"]))
        else:
            st.error(response.text)

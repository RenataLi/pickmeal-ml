# PickMeal service

This folder contains the FastAPI backend and the Streamlit demo application for PickMeal.

## What the service does

The current service supports:

- menu parsing from raw OCR text or uploaded images
- OCR with `PaddleOCR` and `EasyOCR`
- structured dish extraction with the current cascade parser
- nutrition-aware enrichment:
  allergens, ingredient hints, diet flags, calorie ranges, confidence
- recommendation and budget-aware dish combinations
- PostgreSQL + `pgvector` storage for parsed sessions, recommendation runs, and dish embeddings
- similarity search over stored dishes

## Run locally

From the project root:

```bash
pip install -r service/requirements-service.txt
uvicorn service.app.main:app
```

Run Streamlit in a second terminal:

```bash
streamlit run service/streamlit_app.py
```

## Run with Docker

From the project root:

```bash
cp .env.example .env
docker compose up --build
```

This starts:
- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`
- PostgreSQL with `pgvector`

## Important environment variables

- `PICKMEAL_OCR_BACKEND`
- `PICKMEAL_OCR_FALLBACK_BACKEND`
- `PICKMEAL_OCR_LANGS`
- `PICKMEAL_PADDLE_DET_MODEL`
- `PICKMEAL_OCR_MAX_IMAGE_SIDE`
- `PICKMEAL_DATABASE_URL`
- `PICKMEAL_EMBEDDING_DIMENSIONS`
- `PICKMEAL_EMBEDDING_MODEL`

See [.env.example](/Users/renataalieva/Desktop/MDS/pickmeal-ml/.env.example) for the current defaults.

## Main endpoints

### Health and stats

- `GET /health`
- `GET /stats/dataset`
- `GET /stats/parser`
- `GET /stats/line-role`
- `GET /stats/ocr`
- `GET /stats/storage`

### Parsing and recommendation

- `POST /parse/text`
- `POST /parse/image`
- `POST /recommend`

### Storage and retrieval

- `POST /storage/similar`

## Current storage entities

When storage is enabled, the service persists:

- `menu_sessions`
- `parsed_items`
- `dish_embeddings`
- `recommendation_runs`

## Demo flow

The Streamlit demo is designed to show the full product-style pipeline:

1. upload and rotate a menu image;
2. run OCR and parsing;
3. inspect OCR lines and parsed dishes;
4. run recommendation under user constraints;
5. inspect stored session ids, stored recommendation ids, and similar dishes from storage.

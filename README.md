# PickMeal ML

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-cache-DC382D?logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-storage-4169E1?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-similarity-336791)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-OCR-0062B1)

PickMeal is a menu-understanding and dish recommendation service. It turns restaurant menu images into structured dish candidates, enriches them with food-related signals, and ranks dishes under user constraints such as allergens, diet preferences, budget, and approximate calorie targets.

The repository contains the ML pipeline, service runtime, storage and retrieval layer, Redis cache path, and a Streamlit demo for local inspection.

## What PickMeal Does

PickMeal is designed for menus where important information is incomplete. A menu often gives a dish name, section, short description, and price, but not a full ingredient list, verified calories, or reliable allergen labels. The service keeps this uncertainty explicit by storing ranges, confidence values, evidence rows, and caution notes instead of overconfident single-value claims.

The current end-to-end flow is:

1. Upload a restaurant menu image or provide OCR-like text lines.
2. Run OCR and recover ordered text lines with coordinates and confidence values.
3. Classify line roles and parse dishes with names, sections, descriptions, and prices.
4. Enrich parsed dishes with ingredient hints, allergen signals, diet flags, calorie ranges, and confidence.
5. Rank dishes using semantic matching, rules, user constraints, and optional reranking.
6. Store sessions, parsed items, embeddings, and recommendation runs in PostgreSQL.
7. Reuse repeated expensive responses through Redis-backed cache paths.
8. Inspect the pipeline through FastAPI diagnostics and Streamlit screens.

## Main Components

| Component | Purpose |
| --- | --- |
| OCR service | Extracts menu text lines from uploaded images using configurable OCR backends. |
| Line-role model | Labels rows as item names, descriptions, prices, sections, or noise. |
| Parser | Groups OCR rows into structured dish candidates. |
| Enrichment service | Adds ingredient hints, allergens, diet flags, calorie intervals, and confidence notes. |
| Recommendation service | Ranks dishes under user preferences, budget, calorie, diet, and allergen constraints. |
| Storage service | Persists menu sessions, parsed items, embeddings, and recommendation runs. |
| RAG and LLM service | Retrieves supporting evidence and optionally creates bounded dish-card text. |
| Redis cache | Stores repeated recommendation and LLM enrichment responses with TTL and counters. |
| Streamlit demo | Provides an interactive local UI for parsing, recommendations, diagnostics, and model reports. |

## Technology Stack

- **API and runtime:** FastAPI, Pydantic, Uvicorn
- **Demo UI:** Streamlit
- **OCR:** RapidOCR, EasyOCR, PaddleOCR mobile and quality modes
- **ML and ranking:** scikit-learn, TF-IDF, sentence-transformer or FastEmbed fallback, optional CatBoost reranker
- **Storage:** PostgreSQL, pgvector-compatible vector search, JSON embedding fallback
- **Caching:** Redis with TTL, hit/miss/set counters, category-separated keys
- **Packaging:** Docker, Docker Compose
- **Testing:** pytest, FastAPI TestClient

## Repository Structure

```text
service/
  app/
    api/routes/              FastAPI route groups
    resources/               Seed resources for local runtime
    services/                OCR, parser, enrichment, storage, cache, RAG, LLM, recommendation logic
    main.py                  Gateway API application
    schemas.py               Shared request and response models
  scripts/                   Service-level utility and benchmark scripts
  streamlit_app.py           Local demo UI

src/pickmeal_ml/
  data/                      Dataset preparation and validation scripts
  models/                    OCR benchmarks, parser evaluation, line-role training, recommendation models

reports/                     Model, parser, OCR, and demo report artifacts
tests/                       Unit and route-level tests
docker-compose.yml           Local multi-service stack
Dockerfile.api               API and worker service image
Dockerfile.streamlit         Streamlit image
```

## Quick Start With Docker

Docker Compose is the easiest way to run the full local stack.

```bash
cp .env.example .env
docker compose up --build
```

This starts:

- FastAPI gateway: `http://localhost:8000`
- Streamlit demo: `http://localhost:8501`
- PostgreSQL with pgvector on port `5432`
- Redis on port `6379`
- Separate OCR, parser, recommendation, and LLM worker services

Useful checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats/cache
curl http://localhost:8000/stats/storage
curl http://localhost:8000/stats/nutrition
```

## Local Run Without Docker

Use this mode when you want faster iteration on Python code.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r service/requirements-service.txt
```

Start the API:

```bash
uvicorn service.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Start Streamlit in another terminal:

```bash
PICKMEAL_API_URL=http://127.0.0.1:8000 streamlit run service/streamlit_app.py --server.port 8501
```

For storage, Redis, and the full distributed service layout, prefer Docker Compose. Local mode can still run the core parsing and recommendation paths, while optional components depend on the configured environment variables.

## Configuration

Common environment variables are listed in [.env.example](.env.example).

| Variable | Purpose |
| --- | --- |
| `PICKMEAL_API_URL` | API URL used by Streamlit. |
| `PICKMEAL_DATABASE_URL` | PostgreSQL connection string for persistence and retrieval. |
| `PICKMEAL_REDIS_ENABLED` | Enables Redis-backed cache paths. |
| `PICKMEAL_REDIS_URL` | Redis connection string. |
| `PICKMEAL_OCR_BACKEND` | Default OCR backend: `auto`, `rapidocr`, `easyocr`, `paddleocr_mobile`, or `paddleocr_quality`. |
| `PICKMEAL_OCR_FALLBACK_BACKEND` | Fallback OCR backend when the requested one is unavailable. |
| `PICKMEAL_LLM_ENABLED` | Enables OpenAI-compatible LLM enrichment. |
| `PICKMEAL_LLM_BASE_URL` | OpenAI-compatible provider base URL. |
| `PICKMEAL_LLM_MODEL` | LLM model name. |
| `PICKMEAL_RAG_ENABLED` | Enables evidence retrieval for dish-card enrichment. |

## OCR Backends

The Streamlit demo exposes an OCR engine selector. The API accepts the selected backend through `POST /parse/image`.

Supported backend names:

- `auto`
- `rapidocr`
- `easyocr`
- `paddleocr_mobile`
- `paddleocr_quality`

Backend availability depends on the current runtime, installed packages, cached model files, and hardware. The service exposes the active OCR state through:

```bash
curl http://localhost:8000/stats/ocr
```

## API Overview

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Basic API health check. |
| `POST /parse/text` | Parse manually provided OCR-like text lines. |
| `POST /parse/image` | Run OCR, parsing, enrichment, and persistence for an uploaded menu image. |
| `POST /recommend` | Rank parsed dishes under user constraints. |
| `POST /storage/similar` | Retrieve similar stored dishes. |
| `POST /llm/enrich-items` | Generate optional grounded dish-card fields. |
| `GET /stats/cache` | Inspect Redis state, counters, TTLs, and namespace counts. |
| `GET /stats/storage` | Inspect storage initialization and persisted object counts. |
| `GET /stats/nutrition` | Inspect nutrition reference loading and source counts. |
| `GET /stats/runtime` | Inspect route and stage timings. |

## Redis Cache

Redis is used for repeated recommendation and LLM enrichment requests. Cache keys are built from normalized request payloads plus runtime context, so changes in request data, model settings, prompt settings, or retrieval configuration produce a new key.

The cache layer records:

- enabled and available state
- namespace
- recommendation and LLM TTL values
- get, hit, miss, and set counters
- per-category key counts
- last error, when Redis is unavailable

The recommendation cache behavior can be exercised with:

```bash
python service/scripts/benchmark_recommend_cache.py --repeats 40
```

## Validation

Run the test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

The current tests cover:

- Redis key behavior, context changes, TTL expiration, and cache counters
- LLM and recommendation cache route behavior
- confidence-aware enrichment and calorie logic
- recommendation filtering under uncertain calorie estimates
- OCR, parsing, and service health checks

## Current Reports

Selected local reports are stored under [reports](reports):

- OCR backend comparison
- line-role classifier metrics
- parser cascade metrics
- recommendation reranker artifacts
- Streamlit and pre-defense assets

Dataset preparation and gold data utilities live under [src/pickmeal_ml/data](src/pickmeal_ml/data).

## Notes

PickMeal is a local research and demo service, not a medical or nutritional authority. Calorie, diet, and allergen outputs are decision-support signals derived from menu evidence, heuristics, and reference matches. The system intentionally keeps confidence and caution fields visible so users can inspect uncertainty instead of receiving unsupported exact claims.

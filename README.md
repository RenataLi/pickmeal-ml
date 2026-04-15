# pickmeal-ml

PickMeal is a dish recommendation service that starts from restaurant menu images and produces structured dishes, calorie and allergen hints, and recommendation results under user constraints.

This repository now contains both the ML pipeline and the backend demo stack:
- OCR and menu parsing
- gold dataset preparation and evaluation
- FastAPI service
- Streamlit product demo
- PostgreSQL + pgvector storage layer
- Docker packaging

## What the system does

The current end-to-end flow is:

1. a menu image is uploaded to the service;
2. OCR extracts text lines with coordinates;
3. a parser builds dish items from names, prices, sections, and descriptions;
4. an enrichment layer adds ingredient hints, allergen signals, diet flags, and calorie ranges;
5. a recommendation layer ranks dishes and builds budget-aware dish combinations;
6. parsed results and embeddings can be stored in PostgreSQL with `pgvector` for later retrieval.

## Current stack

- `PaddleOCR` as the main OCR backend
- `EasyOCR` as fallback / comparison baseline
- line-role classifier based on `SGDClassifier`
- cascade parser for structured menu item extraction
- FastAPI for the API layer
- Streamlit for the demo UI
- PostgreSQL + `pgvector` for storage and similarity search
- optional OpenAI-compatible LLM enrichment for human-friendly dish cards
- Docker + `docker-compose` for local deployment

## Data sources

- **Roboflow Menu Text Box v3** for layout and detection data
- **Kaggle Indian Restaurant Menu Card Images** as raw menu image corpus
- **USDA FoodData Central** as a nutrition reference source
- **Open Food Facts** as an auxiliary nutrition / ingredient source

## Current dataset status

The annotation and gold-building pipeline has already been expanded beyond the original small baseline:

- `126` menus
- `4209` gold rows
- `9740` OCR rows

These statistics come from:
- [data/processed/gold/build_full_gold_stats.json](/Users/renataalieva/Desktop/MDS/pickmeal-ml/data/processed/gold/build_full_gold_stats.json)

## Current model and system results

### OCR benchmark

From [reports/ocr_backend_benchmark_cascade_v1/ocr_backend_metrics.json](/Users/renataalieva/Desktop/MDS/pickmeal-ml/reports/ocr_backend_benchmark_cascade_v1/ocr_backend_metrics.json):

- `EasyOCR`: `item_f1 = 0.6433`, `price_accuracy = 0.4451`
- `PaddleOCR`: `item_f1 = 0.7034`, `price_accuracy = 0.9524`

### Line-role classifier

From [reports/line_role_expanded_sgd_v1/line_role_metrics.json](/Users/renataalieva/Desktop/MDS/pickmeal-ml/reports/line_role_expanded_sgd_v1/line_role_metrics.json):

- `accuracy = 0.8483`
- `macro_f1 = 0.7023`
- `weighted_f1 = 0.8492`

### Parser

From [reports/parser_cascade_v1_expanded/parser_metrics_test.json](/Users/renataalieva/Desktop/MDS/pickmeal-ml/reports/parser_cascade_v1_expanded/parser_metrics_test.json):

- `item_precision = 0.6244`
- `item_recall = 0.3196`
- `item_f1 = 0.4228`
- `section_accuracy = 0.5119`
- `price_accuracy = 0.3984`

## Repository structure

```text
src/pickmeal_ml/data/
    build_full_gold_jsonl.py
    build_line_role_dataset.py
    build_manifests.py
    import_gpt_annotation_batches.py
    prepare_gpt_annotation_batch.py
    validate_gpt_annotation.py
    ...

src/pickmeal_ml/models/
    benchmark_ocr_backends.py
    evaluate_parser_with_module_layout.py
    line_role_runtime.py
    menu_structuring_baseline_v2.py
    menu_structuring_cascade_v1.py
    train_line_role_classifier.py
    ...

service/app/
    api/routes/
    services/
    main.py
    schemas.py

service/streamlit_app.py
docker-compose.yml
Dockerfile.api
Dockerfile.streamlit
```

## Running the project locally

### API and Streamlit without Docker

```bash
pip install -r service/requirements-service.txt
uvicorn service.app.main:app
streamlit run service/streamlit_app.py
```

### Full stack with Docker

```bash
cp .env.example .env
docker compose up --build
```

This starts:
- API on `http://localhost:8000`
- Streamlit on `http://localhost:8501`
- PostgreSQL with `pgvector`

## Storage layer

The service can persist results to PostgreSQL when `PICKMEAL_DATABASE_URL` is configured.

Current persisted entities:
- `menu_sessions`
- `parsed_items`
- `dish_embeddings`
- `recommendation_runs`

The service also exposes similarity lookup over stored dish embeddings through `pgvector`.

## What is already implemented

- expanded parser gold dataset
- line-role model training
- OCR backend comparison
- parser evaluation
- FastAPI service
- Streamlit demo
- nutrition and allergen enrichment
- recommendation and budget combinations
- PostgreSQL + pgvector persistence
- Docker packaging
- optional LLM-based dish card generation

## What is still future work

- stronger parser models with higher recall
- richer nutrition ingestion and refresh
- LLM-based text enrichment
- more robust normalization and dish catalog building
- monitoring and production-grade logging

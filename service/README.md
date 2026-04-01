## Service possibilities

- exposes a FastAPI service;
- reads parser metrics and dataset stats from the current repository;
- parses OCR text through the existing parser baseline v2;
- optionally runs EasyOCR on uploaded images;
- provides a recommendation baseline using TF-IDF or sentence-transformer embeddings.

## Run the API instructions

From the project root:

```bash
pip install -r service/requirements-service.txt
uvicorn service.app.main:app --reload
```

If you want image upload parsing, also install EasyOCR:

```bash
pip install easyocr
```

## Run the UI

```bash
streamlit run service/streamlit_app.py
```

## Endpoints

- `GET /health`
- `GET /stats/dataset`
- `GET /stats/parser`
- `POST /parse/text`
- `POST /parse/image`
- `POST /recommend`

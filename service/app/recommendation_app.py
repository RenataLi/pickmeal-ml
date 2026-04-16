from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, RecommendRequest, RecommendResponse
from .services.recommendation_service import recommend_items


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} Recommendation Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        engine_used, n_candidates, rows, combo_rows = recommend_items(payload)
        return RecommendResponse(
            recommendation_id=None,
            engine_used=engine_used,
            n_candidates=n_candidates,
            rows=rows,
            combo_rows=combo_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

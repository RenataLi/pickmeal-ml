from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, RecommendRequest, RecommendResponse, RuntimeStatsResponse
from .services.recommendation_service import recommend_items
from .services.runtime_monitoring_service import attach_runtime_monitoring, load_runtime_stats, measure_stage


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} Recommendation Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SERVICE_ID = attach_runtime_monitoring(app, service_id=app.title)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.get("/runtime", response_model=RuntimeStatsResponse)
def runtime_stats() -> RuntimeStatsResponse:
    return RuntimeStatsResponse(**load_runtime_stats(SERVICE_ID))


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    timer = measure_stage(
        SERVICE_ID,
        "recommendation",
        "recommend_items",
        details={"n_items": len(payload.items), "engine_requested": payload.engine},
    )
    try:
        engine_used, n_candidates, rows, combo_rows = recommend_items(payload)
        timer.finish(
            ok=True,
            extra_details={
                "engine_used": engine_used,
                "n_candidates": n_candidates,
                "n_rows": len(rows),
                "n_combo_rows": len(combo_rows),
            },
        )
        return RecommendResponse(
            recommendation_id=None,
            engine_used=engine_used,
            n_candidates=n_candidates,
            rows=rows,
            combo_rows=combo_rows,
        )
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

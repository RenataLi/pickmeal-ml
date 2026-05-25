from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, RecommendRequest, RecommendResponse, RuntimeStatsResponse
from .services.redis_cache_service import get_cached_payload, initialize_cache, set_cached_payload
from .services.recommendation_service import recommend_items, recommendation_cache_context
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


@app.on_event("startup")
def startup() -> None:
    initialize_cache()


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    cache_context = recommendation_cache_context()
    cache_lookup_timer = measure_stage(
        SERVICE_ID,
        "recommendation",
        "cache_lookup",
        details={"n_items": len(payload.items), "engine_requested": payload.engine},
    )
    cached = get_cached_payload("recommendation", payload.model_dump(mode="json"), key_context=cache_context)
    if isinstance(cached, dict):
        cache_lookup_timer.finish(
            ok=True,
            extra_details={
                "cache_hit": True,
                "engine_used": str(cached.get("engine_used", "unknown")),
                "n_candidates": int(cached.get("n_candidates", 0)),
            },
        )
        return RecommendResponse(**cached)
    cache_lookup_timer.finish(ok=True, extra_details={"cache_hit": False})

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
        response = RecommendResponse(
            recommendation_id=None,
            engine_used=engine_used,
            n_candidates=n_candidates,
            rows=rows,
            combo_rows=combo_rows,
        )
        cache_store_timer = measure_stage(
            SERVICE_ID,
            "recommendation",
            "cache_store",
            details={"engine_used": engine_used, "n_items": len(payload.items)},
        )
        stored = set_cached_payload(
            "recommendation",
            payload.model_dump(mode="json"),
            response.model_dump(mode="json"),
            settings.recommendation_cache_ttl_seconds,
            key_context=cache_context,
        )
        cache_store_timer.finish(ok=True, extra_details={"stored": stored})
        return response
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

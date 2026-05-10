from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...config import get_settings
from ...schemas import NutritionRefreshRequest, NutritionRefreshResponse, NutritionStatsResponse
from ...services.nutrition_reference_service import load_nutrition_reference_stats, refresh_nutrition_reference
from ...services.runtime_monitoring_service import measure_stage

router = APIRouter(prefix="/nutrition", tags=["nutrition"])


@router.get("/reference/stats", response_model=NutritionStatsResponse)
def nutrition_reference_stats() -> NutritionStatsResponse:
    try:
        payload = load_nutrition_reference_stats()
        return NutritionStatsResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reference/refresh", response_model=NutritionRefreshResponse)
def nutrition_reference_refresh(payload: NutritionRefreshRequest) -> NutritionRefreshResponse:
    settings = get_settings()
    timer = measure_stage(settings.api_title, "nutrition", "refresh_reference", details={"force": payload.force})
    try:
        result = refresh_nutrition_reference(force=payload.force)
        timer.finish(ok=True, extra_details={"skipped": result.get("skipped"), "record_count": result.get("record_count")})
        return NutritionRefreshResponse(**result)
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

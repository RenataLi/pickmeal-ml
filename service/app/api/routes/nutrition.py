from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import NutritionRefreshRequest, NutritionRefreshResponse, NutritionStatsResponse
from ...services.nutrition_reference_service import load_nutrition_reference_stats, refresh_nutrition_reference

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
    try:
        result = refresh_nutrition_reference(force=payload.force)
        return NutritionRefreshResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

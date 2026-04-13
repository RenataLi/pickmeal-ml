from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import RecommendRequest, RecommendResponse
from ...services.recommendation_service import recommend_items

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        engine_used, n_candidates, rows, combo_rows = recommend_items(payload)
        return RecommendResponse(engine_used=engine_used, n_candidates=n_candidates, rows=rows, combo_rows=combo_rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

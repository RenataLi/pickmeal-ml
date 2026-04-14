from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import RecommendRequest, RecommendResponse
from ...services.recommendation_service import recommend_items
from ...services.storage_service import persist_recommendation_result

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        engine_used, n_candidates, rows, combo_rows = recommend_items(payload)
        recommendation_id = persist_recommendation_result(payload, engine_used, n_candidates, rows, combo_rows)
        return RecommendResponse(
            recommendation_id=recommendation_id,
            engine_used=engine_used,
            n_candidates=n_candidates,
            rows=rows,
            combo_rows=combo_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

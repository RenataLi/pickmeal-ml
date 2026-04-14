from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import SimilarDishesRequest, SimilarDishesResponse
from ...services.storage_service import find_similar_dishes

router = APIRouter(prefix="/storage", tags=["storage"])


@router.post("/similar", response_model=SimilarDishesResponse)
def similar_dishes(payload: SimilarDishesRequest) -> SimilarDishesResponse:
    try:
        rows = find_similar_dishes(payload.query_text, top_k=payload.top_k)
        return SimilarDishesResponse(rows=rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

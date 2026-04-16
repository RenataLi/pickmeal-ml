from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import CombinationRow, RecommendationRow, RecommendRequest, RecommendResponse
from ...services.recommendation_service import recommend_items
from ...services.service_client import post_json, service_urls
from ...services.storage_service import persist_recommendation_result

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _recommend_locally(payload: RecommendRequest) -> RecommendResponse:
    engine_used, n_candidates, rows, combo_rows = recommend_items(payload)
    return RecommendResponse(
        recommendation_id=None,
        engine_used=engine_used,
        n_candidates=n_candidates,
        rows=rows,
        combo_rows=combo_rows,
    )


def _recommend_via_service(payload: RecommendRequest) -> RecommendResponse:
    recommendation_url = service_urls().get("recommendation")
    if not recommendation_url:
        return _recommend_locally(payload)

    raw = post_json(
        recommendation_url,
        "/recommend",
        payload.model_dump(mode="json"),
    )
    return RecommendResponse(
        recommendation_id=None,
        engine_used=str(raw.get("engine_used", "unknown")),
        n_candidates=int(raw.get("n_candidates", 0)),
        rows=[RecommendationRow(**row) for row in raw.get("rows", [])],
        combo_rows=[CombinationRow(**row) for row in raw.get("combo_rows", [])],
    )


@router.post("", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    try:
        response = _recommend_via_service(payload)
        recommendation_id = persist_recommendation_result(
            payload,
            response.engine_used,
            response.n_candidates,
            response.rows,
            response.combo_rows,
        )
        response.recommendation_id = recommendation_id
        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import CombinationRow, RecommendationRow, RecommendRequest, RecommendResponse
from ...services.runtime_monitoring_service import measure_stage
from ...services.recommendation_service import recommend_items
from ...services.service_client import post_json, service_urls
from ...services.storage_service import persist_recommendation_result

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _recommend_locally(payload: RecommendRequest) -> RecommendResponse:
    from ...config import get_settings

    settings = get_settings()
    timer = measure_stage(
        settings.api_title,
        "gateway",
        "recommend_local",
        details={"n_items": len(payload.items), "engine_requested": payload.engine},
    )
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


def _recommend_via_service(payload: RecommendRequest) -> RecommendResponse:
    recommendation_url = service_urls().get("recommendation")
    if not recommendation_url:
        return _recommend_locally(payload)

    from ...config import get_settings

    settings = get_settings()
    timer = measure_stage(
        settings.api_title,
        "gateway",
        "recommendation_service_http",
        details={"n_items": len(payload.items), "engine_requested": payload.engine, "recommendation_url": recommendation_url},
    )
    try:
        raw = post_json(
            recommendation_url,
            "/recommend",
            payload.model_dump(mode="json"),
        )
        timer.finish(
            ok=True,
            extra_details={
                "engine_used": str(raw.get("engine_used", "unknown")),
                "n_candidates": int(raw.get("n_candidates", 0)),
                "n_rows": len(raw.get("rows", [])),
                "n_combo_rows": len(raw.get("combo_rows", [])),
            },
        )
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise
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
        from ...config import get_settings

        settings = get_settings()
        response = _recommend_via_service(payload)
        persist_timer = measure_stage(
            settings.api_title,
            "gateway",
            "persist_recommendation_result",
            details={"n_rows": len(response.rows), "n_combo_rows": len(response.combo_rows)},
        )
        recommendation_id = persist_recommendation_result(
            payload,
            response.engine_used,
            response.n_candidates,
            response.rows,
            response.combo_rows,
        )
        persist_timer.finish(ok=True, extra_details={"recommendation_id": recommendation_id})
        response.recommendation_id = recommendation_id
        return response
    except Exception as exc:
        from ...config import get_settings

        settings = get_settings()
        measure_stage(
            settings.api_title,
            "gateway",
            "recommend",
            details={"n_items": len(payload.items), "engine_requested": payload.engine},
        ).finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

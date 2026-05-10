from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...config import get_settings
from ...schemas import LLMItemEnrichmentRequest, LLMItemEnrichmentResponse, ParsedItem, RAGEvidenceRow
from ...services.llm_service import enrich_items_with_llm
from ...services.runtime_monitoring_service import measure_stage
from ...services.service_client import post_json, service_urls


router = APIRouter(prefix="/llm", tags=["llm"])


def _enrich_locally(payload: LLMItemEnrichmentRequest) -> LLMItemEnrichmentResponse:
    settings = get_settings()
    timer = measure_stage(settings.api_title, "gateway", "llm_enrich_local", details={"n_items": len(payload.items), "top_k": payload.top_k})
    provider_label, model, items, retrieval_rows = enrich_items_with_llm(
        payload.items,
        user_context=payload.user_context,
        top_k=payload.top_k,
    )
    timer.finish(
        ok=True,
        extra_details={"provider_label": provider_label, "model": model, "n_items_returned": len(items), "n_retrieval_rows": len(retrieval_rows)},
    )
    return LLMItemEnrichmentResponse(
        provider_label=provider_label,
        model=model,
        n_items_requested=len(payload.items),
        n_items_returned=len(items),
        items=items,
        retrieval_rows=retrieval_rows,
    )


def _enrich_via_service(payload: LLMItemEnrichmentRequest) -> LLMItemEnrichmentResponse:
    llm_url = service_urls().get("llm")
    if not llm_url:
        return _enrich_locally(payload)

    settings = get_settings()
    timer = measure_stage(settings.api_title, "gateway", "llm_service_http", details={"n_items": len(payload.items), "top_k": payload.top_k, "llm_url": llm_url})
    try:
        raw = post_json(
            llm_url,
            "/llm/enrich-items",
            payload.model_dump(mode="json"),
        )
        timer.finish(
            ok=True,
            extra_details={
                "provider_label": str(raw.get("provider_label", "unknown")),
                "model": str(raw.get("model", "unknown")),
                "n_items_returned": int(raw.get("n_items_returned", len(raw.get("items", [])))),
                "n_retrieval_rows": len(raw.get("retrieval_rows", [])),
            },
        )
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise
    return LLMItemEnrichmentResponse(
        provider_label=str(raw.get("provider_label", "unknown")),
        model=str(raw.get("model", "unknown")),
        n_items_requested=int(raw.get("n_items_requested", len(payload.items))),
        n_items_returned=int(raw.get("n_items_returned", len(raw.get("items", [])))),
        items=[ParsedItem(**row) for row in raw.get("items", [])],
        retrieval_rows=[RAGEvidenceRow(**row) for row in raw.get("retrieval_rows", [])],
    )


@router.post("/enrich-items", response_model=LLMItemEnrichmentResponse)
def enrich_items(payload: LLMItemEnrichmentRequest) -> LLMItemEnrichmentResponse:
    try:
        return _enrich_via_service(payload)
    except Exception as exc:
        settings = get_settings()
        measure_stage(settings.api_title, "gateway", "llm_enrich", details={"n_items": len(payload.items), "top_k": payload.top_k}).finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import LLMItemEnrichmentRequest, LLMItemEnrichmentResponse
from ...services.llm_service import enrich_items_with_llm


router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/enrich-items", response_model=LLMItemEnrichmentResponse)
def enrich_items(payload: LLMItemEnrichmentRequest) -> LLMItemEnrichmentResponse:
    try:
        provider_label, model, items, retrieval_rows = enrich_items_with_llm(
            payload.items,
            user_context=payload.user_context,
            top_k=payload.top_k,
        )
        return LLMItemEnrichmentResponse(
            provider_label=provider_label,
            model=model,
            n_items_requested=len(payload.items),
            n_items_returned=len(items),
            items=items,
            retrieval_rows=retrieval_rows,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

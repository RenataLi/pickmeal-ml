from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, LLMItemEnrichmentRequest, LLMItemEnrichmentResponse
from .services.llm_service import enrich_items_with_llm
from .services.rag_service import initialize_rag


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} LLM Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.on_event("startup")
def startup() -> None:
    initialize_rag()


@app.post("/llm/enrich-items", response_model=LLMItemEnrichmentResponse)
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

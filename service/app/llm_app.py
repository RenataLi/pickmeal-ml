from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, LLMItemEnrichmentRequest, LLMItemEnrichmentResponse, RuntimeStatsResponse
from .services.llm_service import enrich_items_with_llm
from .services.rag_service import initialize_rag
from .services.runtime_monitoring_service import attach_runtime_monitoring, load_runtime_stats, measure_stage


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} LLM Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SERVICE_ID = attach_runtime_monitoring(app, service_id=app.title)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.get("/runtime", response_model=RuntimeStatsResponse)
def runtime_stats() -> RuntimeStatsResponse:
    return RuntimeStatsResponse(**load_runtime_stats(SERVICE_ID))


@app.on_event("startup")
def startup() -> None:
    initialize_rag()


@app.post("/llm/enrich-items", response_model=LLMItemEnrichmentResponse)
def enrich_items(payload: LLMItemEnrichmentRequest) -> LLMItemEnrichmentResponse:
    timer = measure_stage(
        SERVICE_ID,
        "llm",
        "enrich_items",
        details={"n_items": len(payload.items), "top_k": payload.top_k},
    )
    try:
        provider_label, model, items, retrieval_rows = enrich_items_with_llm(
            payload.items,
            user_context=payload.user_context,
            top_k=payload.top_k,
        )
        timer.finish(
            ok=True,
            extra_details={
                "provider_label": provider_label,
                "model": model,
                "n_items_returned": len(items),
                "n_retrieval_rows": len(retrieval_rows),
            },
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
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

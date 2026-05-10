from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, ParseLinesRequest, ParseResponse, RuntimeStatsResponse
from .services.enrichment_service import enrich_items
from .services.line_role_service import line_role_model_is_available, predict_line_roles
from .services.nutrition_reference_service import initialize_nutrition_reference
from .services.parser_service import clean_parsed_items, get_parser_module_name, parse_lines
from .services.runtime_monitoring_service import attach_runtime_monitoring, load_runtime_stats, measure_stage


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} Parser Service", version=settings.api_version)
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
    initialize_nutrition_reference()


@app.post("/parse/lines", response_model=ParseResponse)
def parse_ocr_lines(payload: ParseLinesRequest) -> ParseResponse:
    request_timer = measure_stage(SERVICE_ID, "parser", "parse_request", details={"n_lines": len(payload.ocr_lines)})
    try:
        ocr_lines = payload.ocr_lines
        line_role_timer = measure_stage(SERVICE_ID, "parser", "predict_line_roles", details={"n_lines": len(ocr_lines)})
        line_roles = predict_line_roles(ocr_lines)
        line_role_timer.finish(ok=True, extra_details={"n_roles": len(line_roles)})

        parse_timer = measure_stage(SERVICE_ID, "parser", "parse_lines", details={"n_lines": len(ocr_lines)})
        parsed_items = parse_lines(ocr_lines)
        parse_timer.finish(ok=True, extra_details={"n_candidates": len(parsed_items)})

        enrich_timer = measure_stage(SERVICE_ID, "parser", "enrich_items", details={"n_candidates": len(parsed_items)})
        items = enrich_items(clean_parsed_items(parsed_items, line_roles))
        enrich_timer.finish(ok=True, extra_details={"n_items": len(items)})
        request_timer.finish(ok=True, extra_details={"n_items": len(items)})
        return ParseResponse(
            n_lines=len(ocr_lines),
            n_items=len(items),
            session_id=None,
            ocr_backend=payload.ocr_backend,
            parser_module=settings.parser_module or get_parser_module_name(),
            line_role_model_loaded=line_role_model_is_available(),
            ocr_lines=ocr_lines,
            line_roles=line_roles,
            items=items,
        )
    except Exception as exc:
        request_timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Failed to parse OCR lines: {exc}") from exc

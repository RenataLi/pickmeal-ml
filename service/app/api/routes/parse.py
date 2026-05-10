from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...config import get_settings
from ...schemas import LineRolePrediction, OCRLine, ParseLinesRequest, ParseResponse, ParseTextRequest, ParsedItem
from ...services.enrichment_service import enrich_items
from ...services.line_role_service import line_role_model_is_available, predict_line_roles
from ...services.ocr_service import run_ocr
from ...services.parser_service import clean_parsed_items, get_parser_module_name, parse_lines
from ...services.runtime_monitoring_service import measure_stage
from ...services.service_client import post_json, post_multipart, service_urls
from ...services.storage_service import persist_parse_result

router = APIRouter(prefix="/parse", tags=["parse"])


def _parse_locally(ocr_lines: list[OCRLine], ocr_backend: str | None) -> ParseResponse:
    settings = get_settings()
    service_id = settings.api_title
    line_role_timer = measure_stage(service_id, "gateway", "predict_line_roles_local", details={"n_lines": len(ocr_lines)})
    line_roles = predict_line_roles(ocr_lines)
    line_role_timer.finish(ok=True, extra_details={"n_roles": len(line_roles)})

    parse_timer = measure_stage(service_id, "gateway", "parse_lines_local", details={"n_lines": len(ocr_lines)})
    parsed_items = parse_lines(ocr_lines)
    parse_timer.finish(ok=True, extra_details={"n_candidates": len(parsed_items)})

    enrich_timer = measure_stage(service_id, "gateway", "enrich_items_local", details={"n_candidates": len(parsed_items)})
    items = enrich_items(clean_parsed_items(parsed_items, line_roles))
    enrich_timer.finish(ok=True, extra_details={"n_items": len(items)})
    return ParseResponse(
        n_lines=len(ocr_lines),
        n_items=len(items),
        session_id=None,
        ocr_backend=ocr_backend,
        requested_ocr_backend=ocr_backend,
        parser_module=settings.parser_module or get_parser_module_name(),
        line_role_model_loaded=line_role_model_is_available(),
        ocr_lines=ocr_lines,
        line_roles=line_roles,
        items=items,
    )


def _parse_via_parser_service(ocr_lines: list[OCRLine], ocr_backend: str | None) -> ParseResponse:
    parser_url = service_urls().get("parser")
    if not parser_url:
        return _parse_locally(ocr_lines, ocr_backend=ocr_backend)

    settings = get_settings()
    timer = measure_stage(settings.api_title, "gateway", "parser_service_http", details={"n_lines": len(ocr_lines), "parser_url": parser_url})
    try:
        raw = post_json(
            parser_url,
            "/parse/lines",
            ParseLinesRequest(ocr_lines=ocr_lines, ocr_backend=ocr_backend).model_dump(mode="json"),
        )
        timer.finish(ok=True, extra_details={"n_items": len(raw.get("items", []))})
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise
    return ParseResponse(
        n_lines=int(raw.get("n_lines", len(ocr_lines))),
        n_items=int(raw.get("n_items", len(raw.get("items", [])))),
        session_id=None,
        ocr_backend=raw.get("ocr_backend") or ocr_backend,
        requested_ocr_backend=raw.get("requested_ocr_backend") or ocr_backend,
        ocr_backend_warning=raw.get("ocr_backend_warning"),
        parser_module=raw.get("parser_module"),
        line_role_model_loaded=bool(raw.get("line_role_model_loaded", False)),
        ocr_lines=[OCRLine(**row) for row in raw.get("ocr_lines", [])],
        line_roles=[LineRolePrediction(**row) for row in raw.get("line_roles", [])],
        items=[ParsedItem(**row) for row in raw.get("items", [])],
    )


@router.post("/text", response_model=ParseResponse)
def parse_text(payload: ParseTextRequest) -> ParseResponse:
    ocr_lines: list[OCRLine] = []
    for idx, line in enumerate(payload.lines, start=1):
        if isinstance(line, str):
            ocr_lines.append(OCRLine(text=line, line_order=idx))
        else:
            ocr_lines.append(line)

    settings = get_settings()
    response = _parse_via_parser_service(ocr_lines, ocr_backend="text_input")
    persist_timer = measure_stage(settings.api_title, "gateway", "persist_parse_result", details={"source_kind": "text_input", "n_items": len(response.items)})
    session_id = persist_parse_result(
        source_kind="text_input",
        ocr_backend="text_input",
        parser_module=response.parser_module or get_parser_module_name(),
        ocr_lines=response.ocr_lines,
        line_roles=response.line_roles,
        items=response.items,
    )
    persist_timer.finish(ok=True, extra_details={"session_id": session_id})
    response.session_id = session_id
    response.ocr_backend = "text_input"
    response.requested_ocr_backend = "text_input"
    return response


@router.post("/image", response_model=ParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    langs: str = Form(default="en"),
    backend: str | None = Form(default=None),
) -> ParseResponse:
    settings = get_settings()
    try:
        content = await file.read()
        ocr_url = service_urls().get("ocr")
        if ocr_url:
            ocr_timer = measure_stage(
                settings.api_title,
                "gateway",
                "ocr_service_http",
                details={"requested_backend": backend or "auto", "langs": langs, "ocr_url": ocr_url},
            )
            try:
                ocr_payload = post_multipart(
                    ocr_url,
                    "/ocr/image",
                    files={"file": (file.filename or "menu.png", content, file.content_type or "image/png")},
                    data={"langs": langs, "backend": backend or ""},
                )
                ocr_backend = ocr_payload.get("backend")
                requested_ocr_backend = ocr_payload.get("requested_backend") or backend or ocr_backend
                ocr_backend_warning = ocr_payload.get("warning")
                ocr_lines = [OCRLine(**row) for row in ocr_payload.get("lines", [])]
                ocr_timer.finish(ok=True, extra_details={"backend": ocr_backend, "requested_backend": requested_ocr_backend, "n_lines": len(ocr_lines)})
            except Exception as exc:
                ocr_timer.finish(ok=False, error=str(exc))
                raise
        else:
            local_timer = measure_stage(
                settings.api_title,
                "gateway",
                "ocr_local",
                details={"requested_backend": backend or "auto", "langs": langs},
            )
            try:
                ocr_result = run_ocr(content, langs=langs, backend=backend)
                ocr_backend = ocr_result.backend
                requested_ocr_backend = ocr_result.requested_backend or backend or ocr_backend
                ocr_backend_warning = ocr_result.warning
                ocr_lines = ocr_result.lines
                local_timer.finish(ok=True, extra_details={"backend": ocr_backend, "requested_backend": requested_ocr_backend, "n_lines": len(ocr_lines)})
            except Exception as exc:
                local_timer.finish(ok=False, error=str(exc))
                raise

        response = _parse_via_parser_service(ocr_lines, ocr_backend=ocr_backend)
        persist_timer = measure_stage(settings.api_title, "gateway", "persist_parse_result", details={"source_kind": "image_upload", "n_items": len(response.items)})
        session_id = persist_parse_result(
            source_kind="image_upload",
            ocr_backend=response.ocr_backend,
            parser_module=response.parser_module or get_parser_module_name(),
            ocr_lines=response.ocr_lines,
            line_roles=response.line_roles,
            items=response.items,
        )
        persist_timer.finish(ok=True, extra_details={"session_id": session_id})
        response.session_id = session_id
        response.requested_ocr_backend = requested_ocr_backend
        response.ocr_backend_warning = ocr_backend_warning
        return response
    except RuntimeError as exc:
        measure_stage(settings.api_title, "gateway", "parse_image", details={"requested_backend": backend or "auto", "langs": langs}).finish(ok=False, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        measure_stage(settings.api_title, "gateway", "parse_image", details={"requested_backend": backend or "auto", "langs": langs}).finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...config import get_settings
from ...schemas import OCRLine, ParseResponse, ParseTextRequest
from ...services.enrichment_service import enrich_items
from ...services.line_role_service import line_role_model_is_available, predict_line_roles
from ...services.ocr_service import run_ocr
from ...services.parser_service import clean_parsed_items, get_parser_module_name, parse_lines
from ...services.storage_service import persist_parse_result

router = APIRouter(prefix="/parse", tags=["parse"])


@router.post("/text", response_model=ParseResponse)
def parse_text(payload: ParseTextRequest) -> ParseResponse:
    settings = get_settings()
    lines = payload.lines
    ocr_lines: list[OCRLine] = []
    for idx, line in enumerate(lines, start=1):
        if isinstance(line, str):
            ocr_lines.append(OCRLine(text=line, line_order=idx))
        else:
            ocr_lines.append(line)

    line_roles = predict_line_roles(ocr_lines)
    parsed_items = parse_lines(ocr_lines)
    items = enrich_items(clean_parsed_items(parsed_items, line_roles))
    session_id = persist_parse_result(
        source_kind="text_input",
        ocr_backend="text_input",
        parser_module=settings.parser_module or get_parser_module_name(),
        ocr_lines=ocr_lines,
        line_roles=line_roles,
        items=items,
    )
    return ParseResponse(
        n_lines=len(ocr_lines),
        n_items=len(items),
        session_id=session_id,
        ocr_backend="text_input",
        parser_module=settings.parser_module or get_parser_module_name(),
        line_role_model_loaded=line_role_model_is_available(),
        ocr_lines=ocr_lines,
        line_roles=line_roles,
        items=items,
    )


@router.post("/image", response_model=ParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    langs: str = Form(default="en"),
    backend: str | None = Form(default=None),
) -> ParseResponse:
    settings = get_settings()
    try:
        content = await file.read()
        ocr_result = run_ocr(content, langs=langs, backend=backend)
        ocr_lines = ocr_result.lines
        line_roles = predict_line_roles(ocr_lines)
        parsed_items = parse_lines(ocr_lines)
        items = enrich_items(clean_parsed_items(parsed_items, line_roles))
        session_id = persist_parse_result(
            source_kind="image_upload",
            ocr_backend=ocr_result.backend,
            parser_module=settings.parser_module or get_parser_module_name(),
            ocr_lines=ocr_lines,
            line_roles=line_roles,
            items=items,
        )
        return ParseResponse(
            n_lines=len(ocr_lines),
            n_items=len(items),
            session_id=session_id,
            ocr_backend=ocr_result.backend,
            parser_module=settings.parser_module or get_parser_module_name(),
            line_role_model_loaded=line_role_model_is_available(),
            ocr_lines=ocr_lines,
            line_roles=line_roles,
            items=items,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse image: {exc}") from exc

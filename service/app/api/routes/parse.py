from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...config import get_settings
from ...schemas import OCRLine, ParseResponse, ParseTextRequest
from ...services.line_role_service import line_role_model_is_available, predict_line_roles
from ...services.ocr_service import run_easyocr
from ...services.parser_service import get_parser_module_name, parse_lines

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

    items = parse_lines(ocr_lines)
    line_roles = predict_line_roles(ocr_lines)
    return ParseResponse(
        n_lines=len(ocr_lines),
        n_items=len(items),
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
) -> ParseResponse:
    settings = get_settings()
    try:
        content = await file.read()
        ocr_lines = run_easyocr(content, langs=langs)
        items = parse_lines(ocr_lines)
        line_roles = predict_line_roles(ocr_lines)
        return ParseResponse(
            n_lines=len(ocr_lines),
            n_items=len(items),
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

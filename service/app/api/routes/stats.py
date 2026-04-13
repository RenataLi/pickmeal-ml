from __future__ import annotations

from fastapi import APIRouter

from ...config import get_settings
from ...schemas import DatasetStatsResponse, LineRoleStatsResponse, OCRStatsResponse, ParserStatsResponse
from ...services.stats_service import (
    load_dataset_summary,
    load_line_role_metrics,
    load_line_role_report,
    load_ocr_stats,
    load_parser_comparison,
    load_parser_metrics,
)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/dataset", response_model=DatasetStatsResponse)
def dataset_stats() -> DatasetStatsResponse:
    return DatasetStatsResponse(summary=load_dataset_summary())


@router.get("/parser", response_model=ParserStatsResponse)
def parser_stats() -> ParserStatsResponse:
    settings = get_settings()
    return ParserStatsResponse(
        metrics=load_parser_metrics(),
        active_parser_module=settings.parser_module,
        active_metrics_dir=settings.parser_metrics_dir,
        comparison_rows=load_parser_comparison(),
    )


@router.get("/line-role", response_model=LineRoleStatsResponse)
def line_role_stats() -> LineRoleStatsResponse:
    return LineRoleStatsResponse(
        metrics=load_line_role_metrics(),
        report=load_line_role_report(),
    )


@router.get("/ocr", response_model=OCRStatsResponse)
def ocr_stats() -> OCRStatsResponse:
    payload = load_ocr_stats()
    return OCRStatsResponse(**payload)

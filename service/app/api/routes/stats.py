from __future__ import annotations

from fastapi import APIRouter

from ...schemas import DatasetStatsResponse, ParserStatsResponse
from ...services.stats_service import load_dataset_summary, load_parser_comparison, load_parser_metrics

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/dataset", response_model=DatasetStatsResponse)
def dataset_stats() -> DatasetStatsResponse:
    return DatasetStatsResponse(summary=load_dataset_summary())


@router.get("/parser", response_model=ParserStatsResponse)
def parser_stats() -> ParserStatsResponse:
    return ParserStatsResponse(
        metrics=load_parser_metrics(),
        comparison_rows=load_parser_comparison(),
    )

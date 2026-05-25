from __future__ import annotations

from fastapi import APIRouter

from ...config import get_settings
from ...schemas import CacheStatsResponse, DatasetStatsResponse, LLMStatsResponse, LineRoleStatsResponse, NutritionStatsResponse, OCRStatsResponse, ParserStatsResponse, RAGStatsResponse, RuntimeStatsResponse, StorageStatsResponse
from ...services.stats_service import (
    load_cache_stats,
    load_dataset_summary,
    load_llm_stats,
    load_line_role_metrics,
    load_line_role_report,
    load_nutrition_stats,
    load_ocr_stats,
    load_parser_comparison,
    load_parser_metrics,
    load_rag_stats,
    load_runtime_stats,
    load_storage_stats,
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


@router.get("/storage", response_model=StorageStatsResponse)
def storage_stats() -> StorageStatsResponse:
    payload = load_storage_stats()
    return StorageStatsResponse(**payload)


@router.get("/llm", response_model=LLMStatsResponse)
def llm_stats() -> LLMStatsResponse:
    payload = load_llm_stats()
    return LLMStatsResponse(**payload)


@router.get("/rag", response_model=RAGStatsResponse)
def rag_stats() -> RAGStatsResponse:
    payload = load_rag_stats()
    return RAGStatsResponse(**payload)


@router.get("/nutrition", response_model=NutritionStatsResponse)
def nutrition_stats() -> NutritionStatsResponse:
    payload = load_nutrition_stats()
    return NutritionStatsResponse(**payload)


@router.get("/cache", response_model=CacheStatsResponse)
def cache_stats() -> CacheStatsResponse:
    payload = load_cache_stats()
    return CacheStatsResponse(**payload)


@router.get("/runtime", response_model=RuntimeStatsResponse)
def runtime_stats() -> RuntimeStatsResponse:
    payload = load_runtime_stats()
    return RuntimeStatsResponse(**payload)

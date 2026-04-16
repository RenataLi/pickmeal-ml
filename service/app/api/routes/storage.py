from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...schemas import (
    SimilarDishesRequest,
    SimilarDishesResponse,
    StorageSnapshotImportRequest,
    StorageSnapshotImportResponse,
    StorageSnapshotResponse,
)
from ...services.storage_service import export_storage_snapshot, find_similar_dishes, import_storage_snapshot

router = APIRouter(prefix="/storage", tags=["storage"])


@router.post("/similar", response_model=SimilarDishesResponse)
def similar_dishes(payload: SimilarDishesRequest) -> SimilarDishesResponse:
    try:
        rows = find_similar_dishes(payload.query_text, top_k=payload.top_k)
        return SimilarDishesResponse(rows=rows)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/snapshot", response_model=StorageSnapshotResponse)
def export_snapshot() -> StorageSnapshotResponse:
    try:
        return StorageSnapshotResponse(**export_storage_snapshot())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/snapshot/import", response_model=StorageSnapshotImportResponse)
def import_snapshot(payload: StorageSnapshotImportRequest) -> StorageSnapshotImportResponse:
    try:
        imported_counts = import_storage_snapshot(payload.snapshot)
        return StorageSnapshotImportResponse(imported_counts=imported_counts, replaced_existing=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

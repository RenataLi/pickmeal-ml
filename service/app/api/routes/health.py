from __future__ import annotations

from fastapi import APIRouter

from ...config import get_settings
from ...schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        api_title=settings.api_title,
        api_version=settings.api_version,
    )

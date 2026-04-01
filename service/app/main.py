from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.health import router as health_router
from .api.routes.parse import router as parse_router
from .api.routes.recommend import router as recommend_router
from .api.routes.stats import router as stats_router
from .config import get_settings

settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(stats_router)
app.include_router(parse_router)
app.include_router(recommend_router)

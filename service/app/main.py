from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes.health import router as health_router
from .api.routes.llm import router as llm_router
from .api.routes.nutrition import router as nutrition_router
from .api.routes.parse import router as parse_router
from .api.routes.recommend import router as recommend_router
from .api.routes.storage import router as storage_router
from .api.routes.stats import router as stats_router
from .config import get_settings
from .services.nutrition_reference_service import initialize_nutrition_reference
from .services.rag_service import initialize_rag
from .services.runtime_monitoring_service import attach_runtime_monitoring
from .services.storage_service import initialize_storage

settings = get_settings()

app = FastAPI(title=settings.api_title, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
attach_runtime_monitoring(app, service_id=settings.api_title)

app.include_router(health_router)
app.include_router(stats_router)
app.include_router(nutrition_router)
app.include_router(parse_router)
app.include_router(recommend_router)
app.include_router(storage_router)
app.include_router(llm_router)


@app.on_event("startup")
def startup() -> None:
    initialize_storage()
    initialize_nutrition_reference()
    initialize_rag()

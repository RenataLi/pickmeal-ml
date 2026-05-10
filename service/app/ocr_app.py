from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, OCRImageResponse, RuntimeStatsResponse
from .services.ocr_service import run_ocr
from .services.runtime_monitoring_service import attach_runtime_monitoring, load_runtime_stats, measure_stage


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} OCR Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SERVICE_ID = attach_runtime_monitoring(app, service_id=app.title)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.get("/runtime", response_model=RuntimeStatsResponse)
def runtime_stats() -> RuntimeStatsResponse:
    return RuntimeStatsResponse(**load_runtime_stats(SERVICE_ID))


@app.post("/ocr/image", response_model=OCRImageResponse)
async def ocr_image(
    file: UploadFile = File(...),
    langs: str = Form(default="en"),
    backend: str | None = Form(default=None),
) -> OCRImageResponse:
    timer = measure_stage(SERVICE_ID, "ocr", "run_ocr", details={"requested_backend": backend or "auto", "langs": langs})
    try:
        content = await file.read()
        result = run_ocr(content, langs=langs, backend=backend)
        timer.finish(
            ok=True,
            extra_details={
                "backend": result.backend,
                "requested_backend": result.requested_backend or backend or result.backend,
                "n_lines": len(result.lines),
            },
        )
        return OCRImageResponse(
            backend=result.backend,
            requested_backend=result.requested_backend,
            warning=result.warning,
            n_lines=len(result.lines),
            lines=result.lines,
        )
    except RuntimeError as exc:
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        timer.finish(ok=False, error=str(exc))
        raise HTTPException(status_code=400, detail=f"Failed to run OCR: {exc}") from exc

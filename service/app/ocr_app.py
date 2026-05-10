from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import HealthResponse, OCRImageResponse
from .services.ocr_service import run_ocr


settings = get_settings()

app = FastAPI(title=f"{settings.api_title} OCR Service", version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", api_title=app.title, api_version=settings.api_version)


@app.post("/ocr/image", response_model=OCRImageResponse)
async def ocr_image(
    file: UploadFile = File(...),
    langs: str = Form(default="en"),
    backend: str | None = Form(default=None),
) -> OCRImageResponse:
    try:
        content = await file.read()
        result = run_ocr(content, langs=langs, backend=backend)
        return OCRImageResponse(
            backend=result.backend,
            requested_backend=result.requested_backend,
            warning=result.warning,
            n_lines=len(result.lines),
            lines=result.lines,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to run OCR: {exc}") from exc

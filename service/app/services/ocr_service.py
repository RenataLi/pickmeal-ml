from __future__ import annotations

import io
from functools import lru_cache
from typing import Iterable

import numpy as np
from PIL import Image

from ..config import get_settings
from ..schemas import OCRLine


@lru_cache(maxsize=4)
def _build_reader(langs_key: str):
    try:
        import easyocr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "easyocr is not installed. Install it or use /parse/text."
        ) from exc

    langs = [lang.strip() for lang in langs_key.split(",") if lang.strip()]
    if not langs:
        langs = ["en"]
    return easyocr.Reader(langs, gpu=False)


def run_easyocr(image_bytes: bytes, langs: str | None = None) -> list[OCRLine]:
    settings = get_settings()
    langs_key = langs or settings.default_ocr_langs
    reader = _build_reader(langs_key)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)
    result = reader.readtext(image_array, detail=1, paragraph=False)

    rows: list[OCRLine] = []
    for idx, item in enumerate(result, start=1):
        bbox, text, conf = item
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        rows.append(
            OCRLine(
                text=str(text).strip(),
                line_order=idx,
                ocr_confidence=float(conf),
                bbox_x1=float(min(xs)),
                bbox_y1=float(min(ys)),
                bbox_x2=float(max(xs)),
                bbox_y2=float(max(ys)),
            )
        )
    return rows


def extract_texts(ocr_lines: Iterable[OCRLine]) -> list[str]:
    return [line.text for line in ocr_lines if line.text.strip()]

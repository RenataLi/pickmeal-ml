from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

from ..bootstrap import ensure_src_on_path
from ..config import get_settings
from ..schemas import OCRLine, ParsedItem


ParserFn = Callable[[list[dict]], list[dict]]


def _import_module_by_path(file_path: Path) -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import parser module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_parser_function() -> ParserFn:
    settings = get_settings()
    ensure_src_on_path()

    candidates = [
        settings.parser_module,
        "pickmeal_ml.models.menu_structuring_baseline_v3",
        "pickmeal_ml.models.menu_structuring_baseline_v2",
        "menu_structuring_baseline_v3",
        "menu_structuring_baseline_v2",
    ]

    last_error: Exception | None = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "parse_menu_lines")
        except Exception as exc:  # pragma: no cover - startup fallback path
            last_error = exc

    project_root = settings.project_root
    file_candidates = [
        project_root / "src" / "pickmeal_ml" / "models" / "menu_structuring_baseline_v3.py",
        project_root / "src" / "pickmeal_ml" / "models" / "menu_structuring_baseline_v2.py",
        project_root / "menu_structuring_baseline_v3.py",
        project_root / "menu_structuring_baseline_v2.py",
    ]
    for file_path in file_candidates:
        if file_path.exists():
            module = _import_module_by_path(file_path)
            return getattr(module, "parse_menu_lines")

    raise ImportError(f"Unable to load parser module. Last error: {last_error}")


def get_parser_module_name() -> str:
    settings = get_settings()
    return settings.parser_module


def normalize_input_lines(lines: Iterable[str | OCRLine | dict]) -> list[dict]:
    normalized: list[dict] = []
    for idx, line in enumerate(lines, start=1):
        if isinstance(line, OCRLine):
            payload = line.model_dump()
            payload.setdefault("line_order", idx)
            normalized.append(payload)
        elif isinstance(line, dict):
            payload = dict(line)
            payload.setdefault("line_order", idx)
            if "text" not in payload:
                payload["text"] = str(payload.get("line_text", ""))
            normalized.append(payload)
        else:
            normalized.append({"text": str(line), "line_order": idx})
    return normalized


def parse_lines(lines: Iterable[str | OCRLine | dict]) -> list[ParsedItem]:
    parser = load_parser_function()
    raw_items = parser(normalize_input_lines(lines))
    parsed_items: list[ParsedItem] = []
    for item in raw_items:
        parsed_items.append(ParsedItem(**item))
    return parsed_items

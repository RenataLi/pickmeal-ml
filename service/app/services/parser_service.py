from __future__ import annotations

import importlib
import re
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

from ..bootstrap import ensure_src_on_path
from ..config import get_settings
from ..schemas import LineRolePrediction, OCRLine, ParsedItem


ParserFn = Callable[[list[dict]], list[dict]]
SECTION_LIKE_RE = re.compile(r"^[A-Za-z][A-Za-z&/' +.-]{2,}$")
GENERIC_SECTION_TERMS = {
    "appetizer",
    "appetizers",
    "beverage",
    "beverages",
    "breakfast",
    "dessert",
    "desserts",
    "dinner",
    "drink",
    "drinks",
    "lunch",
    "main course",
    "main courses",
    "menu",
    "pasta",
    "pizza",
    "salad",
    "salads",
    "small plates",
    "soup",
    "soups",
    "soups & salad",
    "soups & salads",
    "special",
    "specials",
    "starter",
    "starters",
}


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


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").strip().split())


def _word_count(text: str) -> int:
    return len([token for token in _normalize_text(text).split(" ") if token])


def _is_section_like(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if normalized.count(",") > 0:
        return False
    if _word_count(normalized) > 6:
        return False
    if not SECTION_LIKE_RE.fullmatch(normalized):
        return False
    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return uppercase_ratio >= 0.55 or normalized.istitle()


def _looks_like_specific_dish_label(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    canonical = re.sub(r"[^a-z]+", " ", normalized.lower()).strip()
    if canonical in GENERIC_SECTION_TERMS:
        return False
    return _is_section_like(normalized)


def _role_lookup(line_roles: Iterable[LineRolePrediction] | None) -> dict[str, LineRolePrediction]:
    lookup: dict[str, LineRolePrediction] = {}
    if not line_roles:
        return lookup
    for role in line_roles:
        normalized = _normalize_text(role.text).lower()
        if not normalized:
            continue
        current = lookup.get(normalized)
        current_score = current.predicted_score if current and current.predicted_score is not None else -1.0
        next_score = role.predicted_score if role.predicted_score is not None else 0.0
        if current is None or next_score >= current_score:
            lookup[normalized] = role
    return lookup


def _repair_misaligned_item(item: ParsedItem) -> ParsedItem:
    dish_name = _normalize_text(item.dish_name)
    description = _normalize_text(item.description)
    section = _normalize_text(item.section)
    has_price = item.price_value is not None or bool(_normalize_text(item.price_text))
    if not section or has_price:
        return item
    if "," not in dish_name:
        return item
    if _word_count(dish_name) < 4:
        return item
    if not _looks_like_specific_dish_label(section):
        return item

    merged_description = " ".join(part for part in [dish_name, description] if part).strip()
    return item.model_copy(
        update={
            "dish_name": section,
            "description": merged_description or None,
            "section": None,
        }
    )


def _should_drop_item(item: ParsedItem, role_lookup: dict[str, LineRolePrediction]) -> bool:
    dish_name = _normalize_text(item.dish_name)
    description = _normalize_text(item.description)
    section = _normalize_text(item.section)
    if not dish_name:
        return True

    role = role_lookup.get(dish_name.lower())
    role_label = role.predicted_label.lower() if role and role.predicted_label else ""
    has_price = item.price_value is not None or bool(_normalize_text(item.price_text))
    has_description = bool(description)
    word_count = _word_count(dish_name)

    if section and dish_name.lower() == section.lower() and not has_price and not has_description:
        return True
    if role_label == "section" and not has_price and not has_description:
        return True
    if role_label == "description" and not has_price and not has_description and ("," in dish_name or word_count >= 5):
        return True
    if not has_price and not has_description and "," in dish_name and word_count >= 4:
        return True
    if _is_section_like(dish_name) and not has_price and not has_description and not section:
        return True
    if not has_price and not has_description and word_count >= 10:
        return True
    return False


def clean_parsed_items(items: list[ParsedItem], line_roles: Iterable[LineRolePrediction] | None = None) -> list[ParsedItem]:
    lookup = _role_lookup(line_roles)
    cleaned: list[ParsedItem] = []
    for item in items:
        item = _repair_misaligned_item(item)
        if _should_drop_item(item, lookup):
            continue
        cleaned.append(item)
    return cleaned


def parse_lines(lines: Iterable[str | OCRLine | dict]) -> list[ParsedItem]:
    parser = load_parser_function()
    raw_items = parser(normalize_input_lines(lines))
    parsed_items: list[ParsedItem] = []
    for item in raw_items:
        parsed_items.append(ParsedItem(**item))
    return parsed_items

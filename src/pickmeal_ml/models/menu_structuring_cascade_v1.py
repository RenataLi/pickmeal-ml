from __future__ import annotations

import re

from pickmeal_ml.models.line_role_runtime import predict_line_roles
from pickmeal_ml.models.menu_structuring_baseline_v2 import parse_menu_lines as parse_baseline_v2


BAD_NAME_TOKENS = {
    "tax",
    "taxes",
    "menu",
    "total",
    "gst",
    "service",
    "charge",
    "hotel",
    "restaurant",
    "mrp",
}


def normalize_key(text: str | None) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return 2.0 * overlap / (len(a_tokens) + len(b_tokens))


def build_item_line_candidates(lines):
    roles = predict_line_roles(lines)
    candidates = []
    for line, role in zip(lines, roles):
        if role.label == "item_name" and (role.score or 0.0) >= 0.35:
            key = normalize_key(line.get("text") if isinstance(line, dict) else getattr(line, "text", ""))
            if key:
                candidates.append(key)
    return candidates


def looks_like_noise_name(name_key: str) -> bool:
    tokens = set(name_key.split())
    if not tokens:
        return True
    if len(tokens) <= 2 and tokens & BAD_NAME_TOKENS:
        return True
    return False


def acceptance_score(item: dict, candidates: list[str]) -> float:
    name_key = normalize_key(item.get("dish_name"))
    if not name_key:
        return 0.0
    return max((similarity(name_key, candidate) for candidate in candidates), default=0.0)


def should_keep_item(item: dict, candidates: list[str]) -> bool:
    name_key = normalize_key(item.get("dish_name"))
    if looks_like_noise_name(name_key):
        return False

    score = acceptance_score(item, candidates)
    has_price = item.get("price_value") is not None or item.get("price_text") is not None
    has_section = item.get("section") is not None
    short_name = len(name_key.split()) <= 2

    if score >= 0.5:
        return True
    if score >= 0.34 and has_price:
        return True
    if score >= 0.28 and short_name and has_section:
        return True
    return False


def should_use_filtered_items(baseline_items: list[dict], filtered_items: list[dict]) -> bool:
    if not filtered_items:
        return False
    if len(baseline_items) < 6:
        return True

    baseline_count = len(baseline_items)
    filtered_count = len(filtered_items)
    if filtered_count < max(4, int(round(0.65 * baseline_count))):
        return False

    baseline_priced = sum(1 for item in baseline_items if item.get("price_value") is not None or item.get("price_text") is not None)
    filtered_priced = sum(1 for item in filtered_items if item.get("price_value") is not None or item.get("price_text") is not None)
    if baseline_priced >= 6 and filtered_priced < max(4, int(round(0.60 * baseline_priced))):
        return False

    return True


def parse_menu_lines(lines):
    baseline_items = parse_baseline_v2(lines)
    has_layout = False
    for line in lines:
        if isinstance(line, dict) and any(line.get(key) is not None for key in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
            has_layout = True
            break
    if not has_layout:
        return baseline_items

    candidates = build_item_line_candidates(lines)
    filtered = []
    for item in baseline_items:
        if should_keep_item(item, candidates):
            score = acceptance_score(item, candidates)
            item = dict(item)
            item["parser_confidence"] = round(max(float(item.get("parser_confidence") or 0.5), min(score + 0.2, 0.95)), 3)
            filtered.append(item)
    if not should_use_filtered_items(baseline_items, filtered):
        return baseline_items
    return filtered

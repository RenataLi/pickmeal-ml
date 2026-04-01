from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from statistics import median
from typing import Iterable, Optional


PRICE_TOKEN_RE = re.compile(r"\d+(?:[.,]\d{1,2})?$")
TRAILING_PRICE_RE = re.compile(
    r"(?:\||:)?\s*(?P<currency>[€$£₹₽])?\s*(?P<value>\d{1,5}(?:[.,]\d{1,2})?)\s*(?P<currency_text>rub|usd|eur|inr|€|\$|£|₹|₽)?\s*$",
    flags=re.IGNORECASE,
)

SECTION_WORDS = {
    "appetizers", "starters", "salads", "soups", "mains", "main course",
    "desserts", "drinks", "beverages", "cocktails", "mocktails", "pizza",
    "pasta", "rice", "biryani", "whisky", "whiskey", "rum", "vodka",
    "gin", "beer", "wine", "brandy", "tequila", "liqueur", "shakes",
    "falooda", "coffee", "tea", "juices", "continental", "indian",
    "punjabi", "chinese", "south indian", "veg", "non veg",
    "sides", "sandwiches", "specials", "entrees", "burgers", "steaks",
    "seafood", "dessert", "classic sandwiches", "primo specials",
}

HEADER_NOISE = {
    "primo", "steak & seafood", "steak and seafood", "menu",
}

WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

BAD_NAME_WORDS = {
    "page", "total", "rate", "price", "menu", "restaurant", "hotel",
    "tax", "gst", "service charge", "mrp",
}

DESCRIPTION_STARTERS = {
    "with", "served", "fresh", "crispy", "grilled", "roasted", "topped", "mixed",
    "includes", "choice", "slow", "juicy", "tender", "field", "baked", "crispy",
    "hand", "crunchy", "creamy", "maple", "house", "smoked",
}

ALLERGEN_KEYWORDS = {
    "milk": {"milk", "cream", "cheese", "butter", "mozzarella", "parmesan", "cheddar", "gouda", "bechamel", "fudge", "gelato"},
    "egg": {"egg", "eggs", "aioli", "mayonnaise", "mayo", "meringue", "brulee"},
    "fish": {"fish", "salmon", "tuna", "cod", "grouper", "walleye", "snapper", "trout", "tilapia"},
    "shellfish": {"shrimp", "prawn", "crab", "lobster", "mussels", "calamari", "scallop", "oyster"},
    "tree_nuts": {"almond", "walnut", "cashew", "hazelnut", "pistachio", "pine nuts", "pecan"},
    "peanut": {"peanut", "peanuts"},
    "wheat": {"wheat", "flour", "bread", "bun", "pasta", "pizza dough", "croutons", "fries"},
    "soy": {"soy", "soybean", "soy sauce", "teriyaki"},
    "sesame": {"sesame", "tahini"},
}


@dataclass
class OCRLine:
    text: str
    line_order: int = 0
    ocr_confidence: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None
    bbox_x2: Optional[float] = None
    bbox_y2: Optional[float] = None
    column: str = "full"

    @property
    def width(self) -> float:
        if self.bbox_x1 is None or self.bbox_x2 is None:
            return 0.0
        return max(0.0, float(self.bbox_x2) - float(self.bbox_x1))

    @property
    def height(self) -> float:
        if self.bbox_y1 is None or self.bbox_y2 is None:
            return 0.0
        return max(0.0, float(self.bbox_y2) - float(self.bbox_y1))

    @property
    def center_x(self) -> float:
        if self.bbox_x1 is None or self.bbox_x2 is None:
            return 0.0
        return (float(self.bbox_x1) + float(self.bbox_x2)) / 2.0

    @property
    def center_y(self) -> float:
        if self.bbox_y1 is None or self.bbox_y2 is None:
            return 0.0
        return (float(self.bbox_y1) + float(self.bbox_y2)) / 2.0


@dataclass
class ColumnState:
    current_section: Optional[str] = None
    pending_name: Optional[str] = None
    pending_desc: list[str] = None

    def __post_init__(self):
        if self.pending_desc is None:
            self.pending_desc = []


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = text.replace("@", "a")
    text = text.replace("°", ".")
    text = text.replace("%", ".")
    text = text.replace("|", " | ")
    text = " ".join(text.strip().split())
    return text


def normalize_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def title_ratio(text: str) -> float:
    tokens = [t for t in normalize_text(text).split() if any(ch.isalpha() for ch in t)]
    if not tokens:
        return 0.0
    ok = 0
    for tok in tokens:
        clean = re.sub(r"[^A-Za-z]+", "", tok)
        if clean and clean[0].isupper():
            ok += 1
    return ok / len(tokens)


def _normalize_currency(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).lower()
    aliases = {"$": "usd", "€": "eur", "£": "gbp", "₹": "inr", "₽": "rub"}
    return aliases.get(value, value)


def _coerce_price_value(raw_text: str, value_text: str) -> Optional[float]:
    raw_text = normalize_text(raw_text)
    value_text = value_text.replace(",", ".")
    try:
        if "." in value_text:
            return float(value_text)
        value_int = int(value_text)
    except Exception:
        return None

    if value_int >= 1000 and value_int < 10000:
        if "|" in raw_text or raw_text.endswith(value_text):
            return round(value_int / 100.0, 2)
    return float(value_int)


def extract_price(text):
    text = normalize_text(text)
    match = TRAILING_PRICE_RE.search(text)
    if match is None:
        return None, None, None
    value = _coerce_price_value(text, match.group("value"))
    currency = _normalize_currency(match.group("currency") or match.group("currency_text"))
    return value, currency, match.group(0).strip()


def has_price(text):
    value, _, _ = extract_price(text)
    return value is not None


def strip_trailing_price(text: str) -> str:
    text = normalize_text(text)
    text = TRAILING_PRICE_RE.sub("", text).strip(" -–|:")
    return normalize_text(text)


def clean_name(text):
    text = strip_trailing_price(text)
    text = re.sub(r"[.]{2,}", " ", text)
    text = text.strip(" -–|:")
    return normalize_text(text)


def is_price_only_line(text):
    text = normalize_text(text)
    if text == "":
        return False
    value, _, _ = extract_price(text)
    return value is not None and clean_name(text) == ""


def find_allergens(text):
    low = normalize_text(text).lower()
    found = []
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        if any(keyword in low for keyword in keywords):
            found.append(allergen)
    return sorted(found)


def is_header_noise(line: OCRLine, page_height: float, median_h: float) -> bool:
    low = normalize_key(line.text)
    if low in HEADER_NOISE:
        return True
    if line.center_y and page_height > 0 and line.center_y < page_height * 0.11:
        if word_count(line.text) <= 4 and not has_price(line.text):
            return True
    if "consuming raw" in low or "foodborne illness" in low:
        return True
    if low.startswith("includes choice"):
        return False
    return False


def is_weekday_line(text: str) -> bool:
    return normalize_key(text) in WEEKDAYS


def is_footer_noise(text: str) -> bool:
    low = normalize_key(text)
    if "foodborne illness" in low or "consuming raw" in low or "risk of" in low:
        return True
    return False


def maybe_section_by_geometry(line: OCRLine, page_width: float, median_h: float) -> bool:
    txt = normalize_text(line.text)
    if txt == "" or has_price(txt):
        return False
    wc = word_count(txt)
    if wc == 0 or wc > 4:
        return False

    tall = median_h > 0 and line.height >= median_h * 1.25
    centered = False
    if page_width > 0 and line.width > 0:
        centered = abs(line.center_x - page_width / 2.0) <= page_width * 0.18 or line.column != "full"

    style_like = title_ratio(txt) >= 0.75 or uppercase_ratio(txt) > 0.6
    return (tall and centered) or (style_like and centered and wc <= 3)


def is_section_line(line: OCRLine, page_width: float, median_h: float) -> bool:
    text = normalize_text(line.text)
    if text == "" or is_weekday_line(text) or is_footer_noise(text):
        return False

    low = normalize_key(text)
    if low in HEADER_NOISE:
        return False

    if low in SECTION_WORDS:
        return True

    if low.endswith("menu") and not has_price(text):
        return True

    return maybe_section_by_geometry(line, page_width, median_h)


def looks_like_description(line: OCRLine) -> bool:
    text = normalize_text(line.text)
    if text == "" or has_price(text) or is_weekday_line(text):
        return False

    low = text.lower()
    wc = word_count(text)

    if wc >= 7:
        return True
    if "," in text or ";" in text:
        return True
    if any(low.startswith(word + " ") for word in DESCRIPTION_STARTERS):
        return True
    if low and low[0].islower() and wc >= 3:
        return True
    return False


def looks_like_item_name(line: OCRLine, next_line: Optional[OCRLine] = None) -> bool:
    text = normalize_text(line.text)
    name = clean_name(text)
    low = normalize_key(name)

    if name == "" or low in BAD_NAME_WORDS or is_weekday_line(text) or is_footer_noise(text):
        return False
    if has_price(text):
        return True
    if word_count(name) > 8:
        return False
    if any(ch.isdigit() for ch in name) and word_count(name) <= 2:
        return False
    if title_ratio(name) < 0.45 and uppercase_ratio(name) < 0.35 and word_count(name) >= 3:
        return False
    if word_count(name) == 1:
        if next_line is not None and is_price_only_line(next_line.text):
            return True
        foodish = {"gelato", "cheesecake", "brownie", "spinach", "asparagus", "corn", "pilaf", "fries", "salad"}
        if low in foodish:
            return True
        return False
    return True


def can_merge(prev: OCRLine, cur: OCRLine, page_width: float) -> bool:
    if normalize_text(prev.text) == "" or normalize_text(cur.text) == "":
        return False
    if has_price(cur.text):
        return False
    if cur.column != prev.column and "full" not in {cur.column, prev.column}:
        return False

    y_gap = 0.0
    if prev.bbox_y2 is not None and cur.bbox_y1 is not None:
        y_gap = float(cur.bbox_y1) - float(prev.bbox_y2)
    same_row = abs(cur.center_y - prev.center_y) <= max(prev.height, cur.height, 10.0) * 0.7
    very_close = y_gap <= max(8.0, prev.height * 0.45 if prev.height > 0 else 8.0)

    cur_wc = word_count(cur.text)
    cur_low = normalize_text(cur.text).lower()
    bridge = cur_wc <= 3 or cur_low.startswith(("or ", "with ", "and ", "of "))
    if not bridge:
        return False

    if not (same_row or very_close):
        return False

    if page_width > 0 and prev.column == "full" and cur.column != "full":
        if cur.center_x < page_width * 0.65:
            return False
    return True


def to_line(obj, idx: int) -> OCRLine:
    if isinstance(obj, OCRLine):
        return obj
    if isinstance(obj, dict):
        return OCRLine(
            text=normalize_text(obj.get("text", obj.get("line_text", ""))),
            line_order=int(obj.get("line_order", idx)),
            ocr_confidence=float(obj["ocr_confidence"]) if obj.get("ocr_confidence") is not None and obj.get("ocr_confidence") == obj.get("ocr_confidence") else None,
            bbox_x1=float(obj["bbox_x1"]) if obj.get("bbox_x1") is not None and obj.get("bbox_x1") == obj.get("bbox_x1") else None,
            bbox_y1=float(obj["bbox_y1"]) if obj.get("bbox_y1") is not None and obj.get("bbox_y1") == obj.get("bbox_y1") else None,
            bbox_x2=float(obj["bbox_x2"]) if obj.get("bbox_x2") is not None and obj.get("bbox_x2") == obj.get("bbox_x2") else None,
            bbox_y2=float(obj["bbox_y2"]) if obj.get("bbox_y2") is not None and obj.get("bbox_y2") == obj.get("bbox_y2") else None,
        )
    return OCRLine(text=normalize_text(obj), line_order=idx)


def assign_columns(lines: list[OCRLine]) -> tuple[list[OCRLine], float, float, float]:
    xs = [line.bbox_x2 for line in lines if line.bbox_x2 is not None]
    ys = [line.bbox_y2 for line in lines if line.bbox_y2 is not None]
    page_width = max(xs) if xs else 0.0
    page_height = max(ys) if ys else 0.0
    heights = [line.height for line in lines if line.height > 0]
    median_h = median(heights) if heights else 24.0

    page_mid = page_width / 2.0 if page_width > 0 else 0.0
    out = []
    for line in lines:
        col = "full"
        if page_width > 0 and line.width > 0:
            full_like = line.width >= page_width * 0.58
            centered_full = line.bbox_x1 is not None and line.bbox_x2 is not None and line.bbox_x1 <= page_width * 0.22 and line.bbox_x2 >= page_width * 0.78
            if full_like or centered_full:
                col = "full"
            else:
                col = "left" if line.center_x < page_mid else "right"
        out.append(replace(line, column=col))
    return out, page_width, page_height, float(median_h)


def merge_short_continuations(lines: list[OCRLine], page_width: float) -> list[OCRLine]:
    if not lines:
        return []
    merged = [lines[0]]
    for line in lines[1:]:
        prev = merged[-1]
        if can_merge(prev, line, page_width):
            new_text = normalize_text(prev.text + " " + line.text)
            merged[-1] = replace(
                prev,
                text=new_text,
                bbox_x1=min(x for x in [prev.bbox_x1, line.bbox_x1] if x is not None),
                bbox_y1=min(y for y in [prev.bbox_y1, line.bbox_y1] if y is not None),
                bbox_x2=max(x for x in [prev.bbox_x2, line.bbox_x2] if x is not None),
                bbox_y2=max(y for y in [prev.bbox_y2, line.bbox_y2] if y is not None),
                ocr_confidence=min(x for x in [prev.ocr_confidence, line.ocr_confidence] if x is not None) if (prev.ocr_confidence is not None or line.ocr_confidence is not None) else None,
            )
        else:
            merged.append(line)
    return merged


def build_full_text(name_text: str, desc_lines: list[str]) -> str:
    parts = [name_text] + list(desc_lines)
    return " ".join([normalize_text(x) for x in parts if normalize_text(x)])


def make_item(item_id, name_text, desc_lines, section_name, column_name):
    full_text = build_full_text(name_text, desc_lines)
    price_value, price_currency, price_text = extract_price(name_text)
    if price_value is None:
        price_value, price_currency, price_text = extract_price(full_text)

    dish_name = clean_name(name_text)
    description_parts = []
    for line in desc_lines:
        clean_line = strip_trailing_price(line)
        if clean_line != "":
            description_parts.append(clean_line)
    description = " ".join(description_parts).strip() or None

    confidence = 0.52
    if price_value is not None:
        confidence += 0.18
    if section_name is not None:
        confidence += 0.12
    if description is not None:
        confidence += 0.08
    if column_name != "full":
        confidence += 0.03
    confidence = min(confidence, 0.95)

    return {
        "local_id": f"item_{item_id:04d}",
        "dish_name": dish_name,
        "description": description,
        "section": section_name,
        "price_value": price_value,
        "price_currency": price_currency,
        "price_text": price_text,
        "explicit_allergens": find_allergens(full_text),
        "parser_confidence": round(confidence, 3),
    }


def finalize_pending(items, item_id, state: ColumnState, global_section: Optional[str], column_name: str):
    if state.pending_name is None:
        return item_id
    section = state.current_section or global_section
    item = make_item(item_id, state.pending_name, state.pending_desc, section, column_name)
    if item["dish_name"] != "" and normalize_key(item["dish_name"]) not in WEEKDAYS:
        items.append(item)
        item_id += 1
    state.pending_name = None
    state.pending_desc = []
    return item_id


def preprocess_lines(lines: Iterable[dict | str | OCRLine]) -> tuple[list[OCRLine], float, float, float]:
    raw = [to_line(obj, idx + 1) for idx, obj in enumerate(lines)]
    raw = [line for line in raw if normalize_text(line.text) != ""]
    raw.sort(key=lambda line: ((line.bbox_y1 if line.bbox_y1 is not None else line.line_order * 100.0), (line.bbox_x1 if line.bbox_x1 is not None else 0.0), line.line_order))
    with_cols, page_width, page_height, median_h = assign_columns(raw)
    merged = merge_short_continuations(with_cols, page_width)
    return merged, page_width, page_height, median_h


def parse_menu_lines(lines):
    prepared, page_width, page_height, median_h = preprocess_lines(lines)

    items = []
    item_id = 1
    global_section = None
    states = {
        "full": ColumnState(),
        "left": ColumnState(),
        "right": ColumnState(),
    }

    for idx, line in enumerate(prepared):
        text = normalize_text(line.text)
        if text == "" or is_footer_noise(text) or is_header_noise(line, page_height, median_h):
            continue

        next_line = prepared[idx + 1] if idx + 1 < len(prepared) else None

        if is_section_line(line, page_width, median_h):
            # full-width section changes the default section for both columns
            if line.column == "full":
                for col_name, state in states.items():
                    item_id = finalize_pending(items, item_id, state, global_section, col_name)
                global_section = clean_name(text) or text
            else:
                state = states[line.column]
                item_id = finalize_pending(items, item_id, state, global_section, line.column)
                state.current_section = clean_name(text) or text
            continue

        if is_weekday_line(text):
            state = states[line.column]
            item_id = finalize_pending(items, item_id, state, global_section, line.column)
            continue

        state = states[line.column]

        if is_price_only_line(text):
            if state.pending_name is not None:
                state.pending_name = normalize_text(f"{state.pending_name} | {text}")
            continue

        if state.pending_name is not None and looks_like_description(line):
            state.pending_desc.append(text)
            continue

        if looks_like_item_name(line, next_line):
            item_id = finalize_pending(items, item_id, state, global_section, line.column)
            state.pending_name = text
            state.pending_desc = []
            continue

        if state.pending_name is not None:
            state.pending_desc.append(text)

    for col_name, state in states.items():
        item_id = finalize_pending(items, item_id, state, global_section, col_name)

    return items


if __name__ == "__main__":
    sample_lines = [
        {"text": "CLASSIC SANDWICHES", "bbox_x1": 300, "bbox_y1": 100, "bbox_x2": 800, "bbox_y2": 150},
        {"text": "Blackened Grouper Sandwich", "bbox_x1": 320, "bbox_y1": 220, "bbox_x2": 700, "bbox_y2": 260},
        {"text": "28", "bbox_x1": 730, "bbox_y1": 220, "bbox_x2": 770, "bbox_y2": 255},
        {"text": "served with remoulade sauce and lettuce", "bbox_x1": 220, "bbox_y1": 260, "bbox_x2": 900, "bbox_y2": 290},
        {"text": "SIDES", "bbox_x1": 180, "bbox_y1": 400, "bbox_x2": 380, "bbox_y2": 450},
        {"text": "Cucumber Salad", "bbox_x1": 100, "bbox_y1": 480, "bbox_x2": 300, "bbox_y2": 510},
        {"text": "8", "bbox_x1": 320, "bbox_y1": 480, "bbox_x2": 340, "bbox_y2": 505},
    ]
    for row in parse_menu_lines(sample_lines):
        print(row)

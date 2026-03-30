import json
import re


PRICE_RE = re.compile(
    r"(?P<currency>[€$₽₹£])?\s?(?P<value>\d{1,5}(?:[.,]\d{1,2})?)\s?(?P<currency_tail>rub|usd|eur|inr)?",
    flags=re.IGNORECASE,
)

SECTION_HINTS = {
    "salads", "soups", "desserts", "pizza", "pasta", "drinks", "tea", "coffee",
    "starters", "mains", "burgers", "rolls", "sushi", "breakfast", "main course",
}

ALLERGEN_KEYWORDS = {
    "milk": {"milk", "cream", "cheese", "butter"},
    "egg": {"egg", "eggs"},
    "fish": {"fish", "salmon", "tuna"},
    "shellfish": {"shrimp", "prawn", "crab", "lobster"},
    "tree_nuts": {"almond", "walnut", "cashew", "hazelnut"},
    "peanut": {"peanut"},
    "wheat": {"wheat", "flour", "bread"},
    "soy": {"soy", "soybean"},
    "sesame": {"sesame", "tahini"},
}


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def extract_price(text):
    text = normalize_text(text)
    match = PRICE_RE.search(text)

    if match is None:
        return None, None

    value_text = match.group("value").replace(",", ".")
    try:
        value = float(value_text)
    except ValueError:
        value = None

    currency = match.group("currency") or match.group("currency_tail")
    if currency is not None:
        currency = currency.strip().lower()

    return value, currency


def remove_price(text):
    text = normalize_text(text)
    return PRICE_RE.sub("", text).strip(" -–|:")


def is_section_line(text):
    text = normalize_text(text)
    text_low = text.lower()

    if len(text_low) == 0:
        return False

    if text_low in SECTION_HINTS:
        return True

    if len(text_low.split()) <= 3 and text_low.isupper():
        return True

    return False


def find_allergens(text):
    text_low = normalize_text(text).lower()
    found = []

    for allergen_name, keywords in ALLERGEN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_low:
                found.append(allergen_name)
                break

    return sorted(found)


def parse_menu_lines(lines):
    items = []
    current_section = None
    current_name = None
    current_desc = []
    item_id = 1

    for line in lines:
        text = normalize_text(line.get("text", ""))
        if text == "":
            continue

        if is_section_line(text):
            if current_name is not None:
                item = make_item(item_id, current_name, current_desc, current_section)
                items.append(item)
                item_id += 1
                current_name = None
                current_desc = []

            current_section = text
            continue

        price_value, price_currency = extract_price(text)
        looks_like_item = price_value is not None or len(text.split()) <= 8

        if current_name is None:
            current_name = text
            continue

        if looks_like_item and len(current_desc) > 0:
            item = make_item(item_id, current_name, current_desc, current_section)
            items.append(item)
            item_id += 1
            current_name = text
            current_desc = []
        else:
            current_desc.append(text)

    if current_name is not None:
        item = make_item(item_id, current_name, current_desc, current_section)
        items.append(item)

    return items


def make_item(item_id, name_text, desc_lines, section_name):
    full_text = " ".join([name_text] + desc_lines)
    price_value, price_currency = extract_price(full_text)

    description = " ".join(remove_price(x) for x in desc_lines).strip()
    if description == "":
        description = None

    confidence = 0.62 if price_value is not None else 0.45

    return {
        "local_id": f"item_{item_id:04d}",
        "dish_name": remove_price(name_text),
        "description": description,
        "section": section_name,
        "price_value": price_value,
        "price_currency": price_currency,
        "explicit_allergens": find_allergens(full_text),
        "source_lines": [name_text] + desc_lines,
        "parser_confidence": confidence,
    }


def group_items_by_section(items):
    groups = {}

    for item in items:
        section_name = item.get("section")
        if section_name is None:
            section_name = "Unsorted"

        if section_name not in groups:
            groups[section_name] = []

        groups[section_name].append(item)

    result = []
    for section_name, section_items in groups.items():
        result.append({
            "section_name": section_name,
            "items": section_items,
        })

    return result


def get_menu_json_schema():
    return {
        "type": "object",
        "properties": {
            "language": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_name": {"type": ["string", "null"]},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dish_name": {"type": "string"},
                                    "description": {"type": ["string", "null"]},
                                    "price_value": {"type": ["number", "null"]},
                                    "price_currency": {"type": ["string", "null"]},
                                    "explicit_allergens": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "dish_name",
                                    "description",
                                    "price_value",
                                    "price_currency",
                                    "explicit_allergens",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["section_name", "items"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["language", "currency", "sections"],
        "additionalProperties": False,
    }


def build_llm_prompt(raw_lines, heuristic_items, language_hint=None):
    raw_text = []
    for line in raw_lines:
        text = normalize_text(line.get("text", ""))
        if text != "":
            raw_text.append(f"- {text}")

    system_prompt = (
        "You are a strict menu parser. "
        "Return valid JSON only. "
        "Do not invent dish names, prices, ingredients, calories, or allergens. "
        "If a field is missing, return null."
    )

    user_prompt = f"""
Language hint: {language_hint or 'unknown'}

Raw OCR lines:
{chr(10).join(raw_text)}

Heuristic parser output:
{json.dumps(heuristic_items, ensure_ascii=False, indent=2)}

Return JSON with this schema:
{json.dumps(get_menu_json_schema(), ensure_ascii=False, indent=2)}
""".strip()

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


if __name__ == "__main__":
    example_lines = [
        {"text": "SALADS"},
        {"text": "Caesar Salad 12.5"},
        {"text": "Chicken, parmesan, croutons"},
        {"text": "Tomato Soup 8"},
        {"text": "Cream, basil"},
    ]

    items = parse_menu_lines(example_lines)
    payload = {
        "language": "en",
        "currency": "usd",
        "sections": group_items_by_section(items),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(json.dumps(build_llm_prompt(example_lines, items, language_hint="en"), ensure_ascii=False, indent=2))

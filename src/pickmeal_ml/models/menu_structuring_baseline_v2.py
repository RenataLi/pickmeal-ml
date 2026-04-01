import re


PRICE_RE = re.compile(
    r"(?P<currency_before>[€$£₹₽])?\s*(?P<value>\d{1,5}(?:[.,]\d{1,2})?)\s*(?P<currency_after>rub|usd|eur|inr|€|\$|£|₹|₽)?",
    flags=re.IGNORECASE,
)

SECTION_WORDS = {
    "appetizers", "starters", "salads", "soups", "mains", "main course",
    "desserts", "drinks", "beverages", "cocktails", "mocktails", "pizza",
    "pasta", "rice", "biryani", "whisky", "whiskey", "rum", "vodka",
    "gin", "beer", "wine", "brandy", "tequila", "liqueur", "shakes",
    "falooda", "coffee", "tea", "juices", "continental", "indian",
    "punjabi", "chinese", "south indian", "veg", "non veg",
}

BAD_NAME_WORDS = {
    "page", "total", "rate", "price", "menu", "restaurant", "hotel",
    "tax", "gst", "service charge", "mrp",
}

ALLERGEN_KEYWORDS = {
    "milk": {"milk", "cream", "cheese", "butter", "mozzarella", "parmesan", "cheddar"},
    "egg": {"egg", "eggs", "aioli", "mayonnaise"},
    "fish": {"fish", "salmon", "tuna", "cod"},
    "shellfish": {"shrimp", "prawn", "crab", "lobster", "mussels", "calamari"},
    "tree_nuts": {"almond", "walnut", "cashew", "hazelnut", "pistachio", "pine nuts"},
    "peanut": {"peanut", "peanuts"},
    "wheat": {"wheat", "flour", "bread", "bun", "pasta", "pizza dough"},
    "soy": {"soy", "soybean", "soy sauce", "teriyaki"},
    "sesame": {"sesame", "tahini"},
}


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = " ".join(text.strip().split())
    return text


def clean_name(text):
    text = normalize_text(text)
    text = PRICE_RE.sub("", text)
    text = re.sub(r"[.]{2,}", " ", text)
    text = text.strip(" -–|:")
    return normalize_text(text)


def extract_price(text):
    text = normalize_text(text)
    match = PRICE_RE.search(text)
    if match is None:
        return None, None, None

    value_text = match.group("value").replace(",", ".")
    try:
        value = float(value_text)
    except Exception:
        value = None

    currency = match.group("currency_before") or match.group("currency_after")
    if currency is not None:
        currency = str(currency).lower()

    return value, currency, match.group(0)


def has_price(text):
    value, _, _ = extract_price(text)
    return value is not None


def is_price_only_line(text):
    text = normalize_text(text)
    if text == "":
        return False
    name = clean_name(text)
    value, _, _ = extract_price(text)
    return value is not None and name == ""


def uppercase_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def is_section_line(text):
    text = normalize_text(text)
    if text == "":
        return False

    low = text.lower()
    no_price = not has_price(text)
    short = len(low.split()) <= 4
    upper_like = uppercase_ratio(text) > 0.65

    if no_price and low in SECTION_WORDS:
        return True

    if no_price and short and upper_like:
        if any(ch.isalpha() for ch in text):
            return True

    if no_price and short and low.endswith("menu"):
        return True

    return False


def looks_like_description(text):
    text = normalize_text(text)
    if text == "":
        return False

    low = text.lower()
    word_count = len(low.split())

    if has_price(text):
        return False

    if "," in text:
        return True

    if word_count >= 5:
        return True

    starters = ["with", "served", "fresh", "crispy", "grilled", "roasted", "topped", "mixed"]
    if any(low.startswith(word) for word in starters):
        return True

    return False


def looks_like_item_name(text):
    text = normalize_text(text)
    if text == "":
        return False

    name = clean_name(text)
    low = name.lower()

    if name == "":
        return False

    if low in BAD_NAME_WORDS:
        return False

    if is_section_line(text):
        return False

    if len(name) < 2:
        return False

    if len(low.split()) > 8 and not has_price(text):
        return False

    letters = sum(ch.isalpha() for ch in name)
    digits = sum(ch.isdigit() for ch in name)
    if letters == 0:
        return False
    if digits > letters:
        return False

    return True


def find_allergens(text):
    low = normalize_text(text).lower()
    found = []
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in low:
                found.append(allergen)
                break
    return sorted(found)


def make_item(item_id, name_text, desc_lines, section_name):
    full_text = " ".join([name_text] + desc_lines)
    price_value, price_currency, price_text = extract_price(full_text)

    dish_name = clean_name(name_text)
    description_parts = []
    for line in desc_lines:
        clean_line = clean_name(line)
        if clean_line != "":
            description_parts.append(clean_line)
    description = " ".join(description_parts).strip()
    if description == "":
        description = None

    confidence = 0.50
    if price_value is not None:
        confidence += 0.20
    if section_name is not None:
        confidence += 0.10
    if description is not None:
        confidence += 0.10
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
        "parser_confidence": confidence,
    }


def finalize_pending(items, item_id, pending_name, pending_desc, current_section):
    if pending_name is None:
        return item_id

    item = make_item(item_id, pending_name, pending_desc, current_section)
    if item["dish_name"] != "":
        items.append(item)
        item_id += 1
    return item_id


def parse_menu_lines(lines):
    normalized_lines = []
    for line in lines:
        if isinstance(line, dict):
            text = normalize_text(line.get("text", ""))
        else:
            text = normalize_text(line)
        if text != "":
            normalized_lines.append(text)

    items = []
    current_section = None
    pending_name = None
    pending_desc = []
    item_id = 1

    for text in normalized_lines:
        if is_section_line(text):
            item_id = finalize_pending(items, item_id, pending_name, pending_desc, current_section)
            pending_name = None
            pending_desc = []
            current_section = clean_name(text) or text
            continue

        if is_price_only_line(text):
            if pending_name is not None:
                pending_name = f"{pending_name} {text}"
            continue

        if has_price(text) and looks_like_item_name(text):
            item_id = finalize_pending(items, item_id, pending_name, pending_desc, current_section)
            pending_name = text
            pending_desc = []
            continue

        if pending_name is None and looks_like_item_name(text):
            pending_name = text
            pending_desc = []
            continue

        if pending_name is not None and looks_like_description(text):
            pending_desc.append(text)
            continue

        if pending_name is not None and looks_like_item_name(text):
            item_id = finalize_pending(items, item_id, pending_name, pending_desc, current_section)
            pending_name = text
            pending_desc = []
            continue

    item_id = finalize_pending(items, item_id, pending_name, pending_desc, current_section)
    return items


if __name__ == "__main__":
    sample_lines = [
        {"text": "MOCKTAILS"},
        {"text": "Fruit Mix Confusions 180"},
        {"text": "fresh fruit, mint, soda"},
        {"text": "Mocha Frappe"},
        {"text": "220"},
        {"text": "coffee, milk, chocolate"},
    ]
    result = parse_menu_lines(sample_lines)
    for row in result:
        print(row)

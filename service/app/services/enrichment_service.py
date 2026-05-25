from __future__ import annotations

import math

from ..schemas import ParsedItem
from .nutrition_reference_service import estimate_from_references


INGREDIENT_KEYWORDS = {
    "chicken": ["chicken"],
    "beef": ["beef", "steak", "burger patty"],
    "pork": ["pork", "bacon", "ham", "salami"],
    "lamb": ["lamb", "mutton"],
    "fish": ["fish", "salmon", "tuna", "cod", "grouper", "trout", "snapper"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "calamari", "mussels", "oyster"],
    "egg": ["egg", "eggs", "omelet", "omelette", "mayo", "mayonnaise", "aioli"],
    "milk": ["milk", "cream", "cheese", "cheddar", "mozzarella", "parmesan", "butter", "paneer", "yogurt", "curd"],
    "wheat": ["wheat", "bread", "bun", "toast", "pasta", "noodles", "pizza", "naan", "roti", "flour", "crouton", "sandwich", "wrap", "burger"],
    "rice": ["rice", "biryani", "pilaf", "pulao"],
    "potato": ["potato", "fries", "chips"],
    "tomato": ["tomato", "marinara", "arrabbiata"],
    "mushroom": ["mushroom"],
    "spinach": ["spinach"],
    "corn": ["corn"],
    "peanut": ["peanut", "groundnut"],
    "tree_nuts": ["almond", "walnut", "cashew", "hazelnut", "pistachio", "pecan"],
    "soy": ["soy", "soy sauce", "tofu", "teriyaki"],
    "sesame": ["sesame", "tahini"],
    "chocolate": ["chocolate", "cocoa", "brownie"],
    "sugar": ["sugar", "caramel", "dessert", "ice cream", "cake", "pastry"],
}

ALLERGEN_MAP = {
    "milk": "milk",
    "egg": "egg",
    "fish": "fish",
    "shellfish": "shellfish",
    "peanut": "peanut",
    "tree_nuts": "tree_nuts",
    "soy": "soy",
    "sesame": "sesame",
    "wheat": "wheat",
}

MEAT_INGREDIENTS = {"chicken", "beef", "pork", "lamb"}
SEAFOOD_INGREDIENTS = {"fish", "shellfish"}
ANIMAL_INGREDIENTS = MEAT_INGREDIENTS | SEAFOOD_INGREDIENTS | {"egg", "milk"}
GLUTEN_INGREDIENTS = {"wheat"}

CATEGORY_RULES = [
    (("salad",), (180, 300, 460)),
    (("soup",), (120, 210, 330)),
    (("sandwich", "burger", "wrap"), (340, 520, 760)),
    (("pizza",), (480, 720, 980)),
    (("pasta", "noodles"), (360, 540, 780)),
    (("rice", "biryani", "pulao", "pilaf"), (340, 520, 760)),
    (("dessert", "cake", "brownie", "ice cream", "gelato"), (240, 380, 620)),
    (("coffee", "tea", "juice", "shake", "smoothie", "drink", "cocktail"), (40, 120, 260)),
    (("seafood", "fish", "steak", "grill"), (260, 430, 700)),
]

CALORIE_ADJUSTMENTS = {
    "milk": (40, 70, 110),
    "egg": (45, 75, 110),
    "wheat": (60, 100, 140),
    "rice": (70, 120, 170),
    "potato": (60, 100, 140),
    "peanut": (60, 95, 140),
    "tree_nuts": (60, 95, 140),
    "beef": (85, 130, 200),
    "pork": (85, 125, 190),
    "lamb": (85, 135, 200),
    "chicken": (70, 110, 165),
    "fish": (60, 100, 150),
    "shellfish": (60, 95, 145),
    "chocolate": (70, 110, 170),
    "sugar": (60, 100, 170),
}

EXPLICIT_DIET_MARKERS = {
    "vegetarian": ["vegetarian", "veggie"],
    "vegan": ["vegan", "plant-based", "plant based"],
    "gluten_free": ["gluten free", "gluten-free", "gf"],
    "dairy_free": ["dairy free", "dairy-free", "lactose free", "lactose-free"],
}

GLUTEN_RISK_TOKENS = ["bread", "bun", "toast", "pasta", "noodles", "pizza", "naan", "roti", "flour", "crouton", "sandwich", "wrap", "burger", "cake", "brownie", "pastry"]
DAIRY_RISK_TOKENS = ["milk", "cream", "cheese", "cheddar", "mozzarella", "parmesan", "butter", "paneer", "yogurt", "curd"]


def _item_text(item: ParsedItem) -> str:
    parts = [item.dish_name or "", item.description or "", item.section or ""]
    return " ".join(part.strip() for part in parts if part).strip().lower()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _field_text(value: str | None) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _has_marker(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def detect_ingredient_evidence(item: ParsedItem) -> dict[str, float]:
    name_text = _field_text(item.dish_name)
    description_text = _field_text(item.description)
    section_text = _field_text(item.section)
    evidence: dict[str, float] = {}

    for ingredient, keywords in INGREDIENT_KEYWORDS.items():
        strength = 0.0
        for keyword in keywords:
            if keyword in name_text:
                strength = max(strength, 1.0)
            if keyword in description_text:
                strength = max(strength, 0.82)
            if keyword in section_text:
                strength = max(strength, 0.58)
        if strength > 0:
            evidence[ingredient] = strength

    ordered = sorted(evidence.items(), key=lambda row: (-row[1], row[0]))
    return {ingredient: strength for ingredient, strength in ordered}


def detect_ingredient_hints(item: ParsedItem) -> list[str]:
    return list(detect_ingredient_evidence(item).keys())


def derive_allergens(existing: list[str], ingredients: list[str]) -> list[str]:
    allergens = {str(x).lower() for x in existing}
    for ingredient in ingredients:
        allergen = ALLERGEN_MAP.get(ingredient)
        if allergen:
            allergens.add(allergen)
    return sorted(allergens)


def derive_diet_flags(ingredients: list[str], allergens: list[str], item: ParsedItem | None = None) -> dict[str, bool]:
    ingredient_set = set(ingredients)
    allergen_set = set(allergens)
    source_text = _item_text(item) if item is not None else " ".join(ingredients).lower()
    explicit_vegetarian = _has_marker(source_text, EXPLICIT_DIET_MARKERS["vegetarian"])
    explicit_vegan = _has_marker(source_text, EXPLICIT_DIET_MARKERS["vegan"])
    explicit_gluten_free = _has_marker(source_text, EXPLICIT_DIET_MARKERS["gluten_free"])
    explicit_dairy_free = _has_marker(source_text, EXPLICIT_DIET_MARKERS["dairy_free"])

    has_positive_evidence = bool(ingredient_set)
    meat_or_seafood_conflict = bool(ingredient_set & (MEAT_INGREDIENTS | SEAFOOD_INGREDIENTS))
    animal_conflict = bool(ingredient_set & ANIMAL_INGREDIENTS) or "milk" in allergen_set
    gluten_conflict = bool(ingredient_set & GLUTEN_INGREDIENTS) or "wheat" in allergen_set
    dairy_conflict = "milk" in allergen_set or "milk" in ingredient_set
    gluten_risk_in_text = any(token in source_text for token in GLUTEN_RISK_TOKENS)
    dairy_risk_in_text = any(token in source_text for token in DAIRY_RISK_TOKENS)

    vegetarian = not meat_or_seafood_conflict and (explicit_vegetarian or explicit_vegan or has_positive_evidence)
    vegan = not animal_conflict and (explicit_vegan or has_positive_evidence)
    gluten_free = not gluten_conflict and (explicit_gluten_free or (has_positive_evidence and not gluten_risk_in_text))
    dairy_free = not dairy_conflict and (explicit_dairy_free or (has_positive_evidence and not dairy_risk_in_text))
    return {
        "vegetarian": vegetarian,
        "vegan": vegan,
        "gluten_free": gluten_free,
        "dairy_free": dairy_free,
    }


def base_calorie_range(item: ParsedItem) -> tuple[float, float, float]:
    text = _item_text(item)
    for keywords, triple in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return triple
    if item.section and "dessert" in item.section.lower():
        return (240, 380, 620)
    if item.section and any(x in item.section.lower() for x in ["drink", "coffee", "tea"]):
        return (40, 120, 260)
    return (220, 360, 560)


def estimate_calories(item: ParsedItem, ingredients: list[str]) -> tuple[float, float, float, float]:
    low, mid, high = base_calorie_range(item)
    for ingredient in ingredients:
        if ingredient in CALORIE_ADJUSTMENTS:
            d_low, d_mid, d_high = CALORIE_ADJUSTMENTS[ingredient]
            low += d_low
            mid += d_mid
            high += d_high
    text = _item_text(item)
    if "fried" in text or "crispy" in text:
        low += 60
        mid += 100
        high += 160
    if "double" in text or "loaded" in text:
        low += 80
        mid += 130
        high += 200
    if "mini" in text or "small" in text:
        low -= 40
        mid -= 50
        high -= 60
    low = max(40.0, low)
    mid = max(low, mid)
    high = max(mid, high)
    confidence = 0.45
    if ingredients:
        confidence += min(0.25, 0.05 * len(ingredients))
    if item.description:
        confidence += 0.15
    if item.section:
        confidence += 0.10
    return round(low, 1), round(mid, 1), round(high, 1), min(confidence, 0.95)


def _reference_blend_weight(
    item: ParsedItem,
    *,
    matched_count: int,
    avg_match_weight: float,
) -> float:
    if matched_count <= 0:
        return 0.0
    parser_confidence = _clamp(float(item.parser_confidence or 0.5), 0.0, 1.0)
    score = (
        -1.15
        + 0.52 * matched_count
        + 0.85 * avg_match_weight
        + 0.22 * float(bool(item.description))
        + 0.18 * float(bool(item.section))
        + 0.35 * parser_confidence
    )
    return round(_clamp(_sigmoid(score), 0.2, 0.92), 3)


def _nutrition_confidence(
    item: ParsedItem,
    ingredient_weights: dict[str, float],
    *,
    matched_count: int,
    avg_match_weight: float,
    heuristic_floor: float,
) -> float:
    parser_confidence = _clamp(float(item.parser_confidence or 0.5), 0.0, 1.0)
    score = (
        -0.7
        + 0.14 * len(ingredient_weights)
        + 0.48 * matched_count
        + 0.75 * avg_match_weight
        + 0.22 * float(bool(item.description))
        + 0.18 * float(bool(item.section))
        + 0.55 * parser_confidence
    )
    calibrated = _sigmoid(score)
    return round(min(0.98, max(float(heuristic_floor), calibrated)), 3)


def estimate_calories_with_references(
    item: ParsedItem,
    ingredients: list[str],
    ingredient_weights: dict[str, float],
) -> tuple[float, float, float, float, list[str]]:
    heur_low, heur_mid, heur_high, heur_confidence = estimate_calories(item, ingredients)
    notes = ["heuristic nutrition estimate from dish title, section and description"]
    reference_estimate = estimate_from_references(ingredients, ingredient_weights=ingredient_weights)
    if not reference_estimate:
        confidence = _nutrition_confidence(
            item,
            ingredient_weights,
            matched_count=0,
            avg_match_weight=0.0,
            heuristic_floor=heur_confidence,
        )
        return heur_low, heur_mid, heur_high, confidence, notes

    ref_low = float(reference_estimate["calories_low"])
    ref_mid = float(reference_estimate["calories_mid"])
    ref_high = float(reference_estimate["calories_high"])
    matched = list(reference_estimate.get("matched_ingredients") or [])
    source_versions = list(reference_estimate.get("source_versions") or [])
    avg_match_weight = float(reference_estimate.get("avg_match_weight") or 0.0)
    blend_weight = _reference_blend_weight(item, matched_count=len(matched), avg_match_weight=avg_match_weight)

    low = round((1.0 - blend_weight) * heur_low + blend_weight * ref_low, 1)
    mid = round((1.0 - blend_weight) * heur_mid + blend_weight * ref_mid, 1)
    high = round((1.0 - blend_weight) * heur_high + blend_weight * ref_high, 1)
    confidence = _nutrition_confidence(
        item,
        ingredient_weights,
        matched_count=len(matched),
        avg_match_weight=avg_match_weight,
        heuristic_floor=min(0.98, heur_confidence + float(reference_estimate.get("confidence_bonus") or 0.0)),
    )

    notes.append("nutrition reference lookup matched ingredient-level records in PostgreSQL")
    if matched:
        notes.append(f"reference-backed estimate used matched ingredients: {', '.join(matched)}")
    if source_versions:
        notes.append(f"nutrition source versions: {', '.join(source_versions)}")
    notes.append(f"reference blend weight: {blend_weight}")
    return low, mid, high, confidence, notes


def enrich_item(item: ParsedItem) -> ParsedItem:
    ingredient_weights = detect_ingredient_evidence(item)
    ingredients = list(ingredient_weights.keys())
    allergens = derive_allergens(item.explicit_allergens, ingredients)
    diet_flags = derive_diet_flags(ingredients, allergens, item=item)
    calories_low, calories_mid, calories_high, nutrition_confidence, notes = estimate_calories_with_references(item, ingredients, ingredient_weights)

    if ingredients:
        notes.append("ingredient hints detected from menu text")
    else:
        notes.append("no strong ingredient hints detected, dietary flags remain conservative")
    if not item.description:
        notes.append("description missing, estimate may be broad")

    return item.model_copy(
        update={
            "ingredient_hints": ingredients,
            "explicit_allergens": allergens,
            "diet_flags": diet_flags,
            "calories_low": calories_low,
            "calories_mid": calories_mid,
            "calories_high": calories_high,
            "nutrition_confidence": round(nutrition_confidence, 3),
            "enrichment_notes": notes,
        }
    )


def enrich_items(items: list[ParsedItem]) -> list[ParsedItem]:
    return [enrich_item(item) for item in items]

from __future__ import annotations

from ..schemas import ParsedItem


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


def _item_text(item: ParsedItem) -> str:
    parts = [item.dish_name or "", item.description or "", item.section or ""]
    return " ".join(part.strip() for part in parts if part).strip().lower()


def detect_ingredient_hints(item: ParsedItem) -> list[str]:
    text = _item_text(item)
    found = []
    for ingredient, keywords in INGREDIENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            found.append(ingredient)
    return found


def derive_allergens(existing: list[str], ingredients: list[str]) -> list[str]:
    allergens = {str(x).lower() for x in existing}
    for ingredient in ingredients:
        allergen = ALLERGEN_MAP.get(ingredient)
        if allergen:
            allergens.add(allergen)
    return sorted(allergens)


def derive_diet_flags(ingredients: list[str], allergens: list[str]) -> dict[str, bool]:
    ingredient_set = set(ingredients)
    allergen_set = set(allergens)
    vegetarian = not bool(ingredient_set & (MEAT_INGREDIENTS | SEAFOOD_INGREDIENTS))
    vegan = vegetarian and not bool(ingredient_set & {"milk", "egg", "honey"})
    gluten_free = not bool(ingredient_set & GLUTEN_INGREDIENTS) and "wheat" not in allergen_set
    dairy_free = "milk" not in allergen_set and "milk" not in ingredient_set
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


def enrich_item(item: ParsedItem) -> ParsedItem:
    ingredients = detect_ingredient_hints(item)
    allergens = derive_allergens(item.explicit_allergens, ingredients)
    diet_flags = derive_diet_flags(ingredients, allergens)
    calories_low, calories_mid, calories_high, nutrition_confidence = estimate_calories(item, ingredients)

    notes = ["heuristic nutrition estimate from dish title, section and description"]
    if ingredients:
        notes.append("ingredient hints detected from menu text")
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

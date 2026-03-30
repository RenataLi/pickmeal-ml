import json
import math

import numpy as np


try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None


def build_item_text(item):
    parts = [item.get("dish_name", "")]

    if item.get("description"):
        parts.append(item["description"])

    if item.get("section"):
        parts.append("section: " + item["section"])

    allergens = item.get("explicit_allergens", [])
    if allergens:
        parts.append("allergens: " + ", ".join(allergens))

    return " | ".join(parts)


def build_query(profile):
    include_text = " ".join(profile.get("include_text", []))
    exclude_text = " ".join("not " + x for x in profile.get("exclude_text", []))
    query = (include_text + " " + exclude_text).strip()

    if query == "":
        query = "safe suitable food"

    return query


def encode_texts(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    if SentenceTransformer is not None:
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float)

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return matrix.toarray().astype(float)

    vocab = sorted({token.lower() for text in texts for token in text.split()})
    matrix = np.zeros((len(texts), len(vocab)), dtype=float)
    token_to_index = {token: i for i, token in enumerate(vocab)}

    for row_id, text in enumerate(texts):
        for token in text.lower().split():
            matrix[row_id, token_to_index[token]] += 1.0

        norm = np.linalg.norm(matrix[row_id])
        if norm > 0:
            matrix[row_id] = matrix[row_id] / norm

    return matrix


def cosine_matrix(a, b):
    if cosine_similarity is not None:
        return cosine_similarity(a, b)

    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def score_item(item, semantic_score, profile):
    score = float(semantic_score)
    reasons = []

    item_allergens = [x.lower() for x in item.get("explicit_allergens", [])]
    blocked_allergens = [x.lower() for x in profile.get("forbidden_allergens", [])]

    overlap = sorted(set(item_allergens) & set(blocked_allergens))
    if overlap:
        return -math.inf, ["hard_filter: allergen_match=" + ",".join(overlap)]

    item_text = build_item_text(item).lower()

    for token in profile.get("exclude_text", []):
        if token.lower() in item_text:
            score -= 0.35
            reasons.append("penalty_exclude_text:" + token)

    for token in profile.get("include_text", []):
        if token.lower() in item_text:
            score += 0.15
            reasons.append("bonus_include_text:" + token)

    max_calories = profile.get("max_calories")
    if max_calories is not None and item.get("calories_mid") is not None:
        if item["calories_mid"] > max_calories:
            score -= 0.50
            reasons.append(f"penalty_calories>{max_calories}")
        else:
            score += 0.10
            reasons.append("bonus_calorie_fit")

    max_price = profile.get("max_price")
    if max_price is not None and item.get("price_value") is not None:
        if item["price_value"] > max_price:
            score -= 0.40
            reasons.append(f"penalty_price>{max_price}")
        else:
            score += 0.08
            reasons.append("bonus_budget_fit")

    return score, reasons


def rank_menu_items(items, profile, top_k=5):
    items = list(items)
    if len(items) == 0:
        return []

    query = build_query(profile)
    item_texts = [build_item_text(item) for item in items]
    texts = [query] + item_texts
    vectors = encode_texts(texts)

    query_vector = vectors[:1]
    item_vectors = vectors[1:]
    semantic_scores = cosine_matrix(query_vector, item_vectors).flatten()

    ranked = []
    for item, semantic_score in zip(items, semantic_scores):
        final_score, reasons = score_item(item, semantic_score, profile)
        if final_score == -math.inf:
            continue

        ranked.append({
            "item": item,
            "semantic_score": float(semantic_score),
            "final_score": float(final_score),
            "reasons": reasons,
        })

    ranked.sort(key=lambda row: row["final_score"], reverse=True)
    return ranked[:top_k]


if __name__ == "__main__":
    items = [
        {
            "item_id": "1",
            "dish_name": "Caesar salad",
            "description": "chicken, parmesan, croutons",
            "section": "Salads",
            "explicit_allergens": ["milk", "wheat"],
            "calories_mid": 520,
            "price_value": 12.5,
        },
        {
            "item_id": "2",
            "dish_name": "Tomato soup",
            "description": "basil, cream",
            "section": "Soups",
            "explicit_allergens": ["milk"],
            "calories_mid": 260,
            "price_value": 8.0,
        },
        {
            "item_id": "3",
            "dish_name": "Grilled vegetables",
            "description": "zucchini, bell pepper, eggplant",
            "section": "Mains",
            "explicit_allergens": [],
            "calories_mid": 190,
            "price_value": 9.5,
        },
    ]

    profile = {
        "include_text": ["light", "vegetables"],
        "exclude_text": ["cheese"],
        "forbidden_allergens": ["peanut", "sesame"],
        "max_calories": 400,
        "max_price": 10,
    }

    result = rank_menu_items(items, profile, top_k=3)
    print(json.dumps(result, ensure_ascii=False, indent=2))

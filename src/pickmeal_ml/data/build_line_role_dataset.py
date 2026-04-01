from pathlib import Path
import argparse
import json
import math
import re

import pandas as pd


PRICE_ONLY_RE = re.compile(r"^[€$£₹₽]?\s*\d{1,5}(?:[.,]\d{1,2})?\s*(?:rub|usd|eur|inr|€|\$|£|₹|₽)?$", re.IGNORECASE)


def resolve_project_root():
    root = Path.cwd().resolve()
    if (root / "data").exists():
        return root
    if (root.parent / "data").exists():
        return root.parent
    raise FileNotFoundError("Project root with data/ was not found.")


def normalize_text(text) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = " ".join(text.strip().split())
    return text


def normalize_key(text) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    return " ".join(text.split())


def token_set(text: str) -> set[str]:
    return set(normalize_key(text).split())


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def looks_like_price_only(text: str) -> bool:
    return bool(PRICE_ONLY_RE.match(normalize_text(text)))


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def uppercase_ratio(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def title_ratio(text: str) -> float:
    tokens = [t for t in normalize_text(text).split() if any(ch.isalpha() for ch in t)]
    if not tokens:
        return 0.0
    good = 0
    for tok in tokens:
        tok = re.sub(r"[^A-Za-z]+", "", tok)
        if tok and tok[0].isupper():
            good += 1
    return good / len(tokens)


def numeric_features(line: dict) -> dict:
    text = normalize_text(line.get("text"))
    x1 = line.get("bbox_x1")
    y1 = line.get("bbox_y1")
    x2 = line.get("bbox_x2")
    y2 = line.get("bbox_y2")
    width = float(x2 - x1) if x1 is not None and x2 is not None else 0.0
    height = float(y2 - y1) if y1 is not None and y2 is not None else 0.0
    center_x = (float(x1) + float(x2)) / 2.0 if x1 is not None and x2 is not None else 0.0
    center_y = (float(y1) + float(y2)) / 2.0 if y1 is not None and y2 is not None else 0.0
    return {
        "word_count": word_count(text),
        "char_count": len(text),
        "digit_count": sum(ch.isdigit() for ch in text),
        "uppercase_ratio": round(uppercase_ratio(text), 6),
        "title_ratio": round(title_ratio(text), 6),
        "comma_count": text.count(","),
        "has_price_token": int(bool(re.search(r"\d", text))),
        "price_only_flag": int(looks_like_price_only(text)),
        "ocr_confidence": float(line.get("ocr_confidence") or 0.0),
        "bbox_width": round(width, 4),
        "bbox_height": round(height, 4),
        "bbox_center_x": round(center_x, 4),
        "bbox_center_y": round(center_y, 4),
        "line_order": int(line.get("line_order") or 0),
    }


def label_line(line_text: str, sections: list[str], dish_names: list[str], descriptions: list[str]) -> str:
    key = normalize_key(line_text)
    if key == "":
        return "noise"
    if looks_like_price_only(line_text):
        return "price"

    for section in sections:
        skey = normalize_key(section)
        if skey and (key == skey or jaccard(line_text, section) >= 0.8):
            return "section"

    for name in dish_names:
        nkey = normalize_key(name)
        if not nkey:
            continue
        if key == nkey:
            return "item_name"
        if key in nkey and word_count(line_text) >= 2:
            return "item_name"
        if jaccard(line_text, name) >= 0.82:
            return "item_name"

    for desc in descriptions:
        dkey = normalize_key(desc)
        if not dkey:
            continue
        if key in dkey and word_count(line_text) >= 3:
            return "description"
        if dkey in key and word_count(desc) >= 3:
            return "description"
        if jaccard(line_text, desc) >= 0.55 and word_count(line_text) >= 3:
            return "description"

    return "noise"


def build_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        menu_id = record["menu_id"]
        page_id = record.get("page_id", "p1")
        items = record.get("items", [])
        sections = [x.get("section") for x in items if x.get("section")]
        dish_names = [x.get("dish_name") for x in items if x.get("dish_name")]
        descriptions = [x.get("description") for x in items if x.get("description")]

        for line in record.get("ocr_lines", []):
            text = normalize_text(line.get("text"))
            feats = numeric_features(line)
            label = label_line(text, sections, dish_names, descriptions)
            row = {
                "menu_id": menu_id,
                "page_id": page_id,
                "text": text,
                "label": label,
            }
            row.update(feats)
            rows.append(row)
    return rows


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(input_path: Path, out_path: Path):
    records = load_jsonl(input_path)
    rows = build_rows(records)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    stats = df["label"].value_counts().sort_index().to_dict()
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Saved:", stats_path)
    print("Rows:", len(df))
    print("Label stats:", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/processed/gold/menu_gold_all_batches_layout.jsonl")
    parser.add_argument("--out_path", type=str, default="data/processed/gold/line_role_dataset.csv")
    args = parser.parse_args()

    project_root = resolve_project_root()
    main(project_root / args.input_path, project_root / args.out_path)

from pathlib import Path
import argparse
import ast
import json

import pandas as pd


def normalize_text(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    return text


def parse_list_value(value):
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]

    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def main(gold_csv_path, ocr_csv_path, out_path):
    gold_df = pd.read_csv(gold_csv_path)
    ocr_df = pd.read_csv(ocr_csv_path)

    gold_df = gold_df.copy()
    ocr_df = ocr_df.copy()

    gold_df["menu_id"] = gold_df["menu_id"].apply(normalize_text)
    gold_df["page_id"] = gold_df["page_id"].apply(normalize_text)

    ocr_df["menu_id"] = ocr_df["menu_id"].apply(normalize_text)
    ocr_df["page_id"] = ocr_df["page_id"].apply(normalize_text)

    records = []

    grouped = gold_df.groupby(["menu_id", "page_id"], dropna=False)

    for (menu_id, page_id), group in grouped:
        ocr_part = ocr_df[
            (ocr_df["menu_id"] == menu_id) &
            (ocr_df["page_id"] == page_id)
        ].copy()

        if "line_order" in ocr_part.columns:
            ocr_part = ocr_part.sort_values("line_order")

        ocr_lines = []
        for _, row in ocr_part.iterrows():
            line_text = normalize_text(row.get("line_text"))
            if line_text is not None:
                ocr_lines.append(line_text)

        items = []
        for _, row in group.iterrows():
            items.append(
                {
                    "item_id": normalize_text(row.get("item_id")),
                    "dish_name": normalize_text(row.get("dish_name")),
                    "section": normalize_text(row.get("section")),
                    "price_text": normalize_text(row.get("price_text")),
                    "price_value": row.get("price_value") if not pd.isna(row.get("price_value")) else None,
                    "currency": normalize_text(row.get("currency")),
                    "description": normalize_text(row.get("description")),
                    "explicit_ingredients": parse_list_value(row.get("explicit_ingredients")),
                    "explicit_allergens": parse_list_value(row.get("explicit_allergens")),
                }
            )

        record = {
            "menu_id": menu_id,
            "page_id": page_id,
            "ocr_lines": ocr_lines,
            "items": items,
        }

        records.append(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved:", out_path)
    print("Menus:", len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv_path", type=str, required=True)
    parser.add_argument("--ocr_csv_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="data/processed/gold/menu_gold_batch_001_final.jsonl")
    args = parser.parse_args()

    main(
        gold_csv_path=Path(args.gold_csv_path),
        ocr_csv_path=Path(args.ocr_csv_path),
        out_path=Path(args.out_path),
    )
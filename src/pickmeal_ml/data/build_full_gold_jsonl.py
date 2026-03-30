from pathlib import Path
import argparse
import ast
import json

import pandas as pd


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


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


def find_ocr_files(search_dir):
    files = sorted(search_dir.rglob("batch_*_ocr.csv"))
    return files


def load_all_ocr(search_dir):
    files = find_ocr_files(search_dir)

    if len(files) == 0:
        raise FileNotFoundError("No OCR CSV files were found.")

    frames = []
    for path in files:
        df = pd.read_csv(path).copy()
        df["source_file"] = path.name
        frames.append(df)

    ocr_df = pd.concat(frames, ignore_index=True)

    return ocr_df, files


def main(gold_csv_path, ocr_search_dir, out_path):
    gold_df = pd.read_csv(gold_csv_path).copy()
    ocr_df, ocr_files = load_all_ocr(ocr_search_dir)

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
            item = {
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
            items.append(item)

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

    stats_path = out_path.parent / "build_full_gold_stats.json"
    stats = {
        "n_menus": len(records),
        "n_gold_rows": int(len(gold_df)),
        "n_ocr_rows": int(len(ocr_df)),
        "n_ocr_files": len(ocr_files),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Saved:", stats_path)
    print("Menus:", len(records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_csv_path",
        type=str,
        default="data/processed/gold/menu_gold_all_batches.csv",
        help="Merged gold CSV",
    )
    parser.add_argument(
        "--ocr_search_dir",
        type=str,
        default="data/processed/annotation_batches",
        help="Directory with batch OCR CSV files",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/gold/menu_gold_all_batches.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        gold_csv_path=project_root / args.gold_csv_path,
        ocr_search_dir=project_root / args.ocr_search_dir,
        out_path=project_root / args.out_path,
    )

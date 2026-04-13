from pathlib import Path
import argparse
import importlib
import json
import sys

import pandas as pd


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def normalize_price(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace(",", ".").strip()
    try:
        return round(float(value), 2)
    except Exception:
        return None


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_gold_map(items):
    gold_map = {}
    for item in items:
        key = normalize_text(item.get("dish_name"))
        if key == "":
            continue
        gold_map.setdefault(key, []).append(item)
    return gold_map


def select_layout_records(layout_records, split_records):
    keys = {
        (
            str(row["menu_id"]).strip(),
            str(row.get("page_id", "p1")).strip(),
        )
        for row in split_records
    }
    selected = []
    for record in layout_records:
        key = (
            str(record["menu_id"]).strip(),
            str(record.get("page_id", "p1")).strip(),
        )
        if key in keys:
            selected.append(record)
    return selected


def evaluate_records(records, parse_menu_lines):
    item_tp = 0
    item_fp = 0
    item_fn = 0
    section_ok = 0
    section_total = 0
    price_ok = 0
    price_total = 0
    description_ok = 0
    description_total = 0
    error_rows = []

    for record in records:
        menu_id = record["menu_id"]
        page_id = record.get("page_id", "p1")
        pred_items = parse_menu_lines(record["ocr_lines"])
        gold_items = record["items"]
        gold_map = build_gold_map(gold_items)
        matched_gold = set()

        for pred in pred_items:
            pred_name = normalize_text(pred.get("dish_name"))
            if pred_name == "":
                item_fp += 1
                error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "empty_pred_name", "pred_name": None, "gold_name": None})
                continue

            gold_candidates = gold_map.get(pred_name, [])
            match_index = None
            for i, _ in enumerate(gold_candidates):
                gold_key = f"{pred_name}__{i}"
                if gold_key not in matched_gold:
                    match_index = i
                    matched_gold.add(gold_key)
                    break

            if match_index is None:
                item_fp += 1
                error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "false_positive_item", "pred_name": pred.get("dish_name"), "gold_name": None})
                continue

            item_tp += 1
            gold_item = gold_candidates[match_index]

            gold_section = normalize_text(gold_item.get("section"))
            pred_section = normalize_text(pred.get("section"))
            if gold_section != "":
                section_total += 1
                if gold_section == pred_section:
                    section_ok += 1
                else:
                    error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "wrong_section", "pred_name": pred.get("dish_name"), "gold_name": gold_item.get("dish_name")})

            gold_price = normalize_price(gold_item.get("price_value"))
            pred_price = normalize_price(pred.get("price_value"))
            if gold_price is not None:
                price_total += 1
                if pred_price == gold_price:
                    price_ok += 1
                else:
                    error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "wrong_price", "pred_name": pred.get("dish_name"), "gold_name": gold_item.get("dish_name")})

            gold_desc = normalize_text(gold_item.get("description"))
            pred_desc = normalize_text(pred.get("description"))
            if gold_desc != "":
                description_total += 1
                if gold_desc == pred_desc:
                    description_ok += 1
                else:
                    error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "wrong_description", "pred_name": pred.get("dish_name"), "gold_name": gold_item.get("dish_name")})

        for key, values in gold_map.items():
            for i, gold in enumerate(values):
                gold_key = f"{key}__{i}"
                if gold_key not in matched_gold:
                    item_fn += 1
                    error_rows.append({"menu_id": menu_id, "page_id": page_id, "error_type": "missed_gold_item", "pred_name": None, "gold_name": gold.get("dish_name")})

    precision = item_tp / (item_tp + item_fp) if (item_tp + item_fp) > 0 else 0.0
    recall = item_tp / (item_tp + item_fn) if (item_tp + item_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "n_menus": len(records),
        "item_precision": round(precision, 4),
        "item_recall": round(recall, 4),
        "item_f1": round(f1, 4),
        "section_accuracy": round(section_ok / section_total, 4) if section_total > 0 else None,
        "price_accuracy": round(price_ok / price_total, 4) if price_total > 0 else None,
        "description_exact_match": round(description_ok / description_total, 4) if description_total > 0 else None,
        "item_tp": item_tp,
        "item_fp": item_fp,
        "item_fn": item_fn,
    }
    return metrics, pd.DataFrame(error_rows)


def main(layout_path, split_reference_path, out_dir, split_name, parser_module):
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    module = importlib.import_module(parser_module)
    parse_menu_lines = getattr(module, "parse_menu_lines")

    layout_records = load_jsonl(layout_path)
    split_records = load_jsonl(split_reference_path)
    records = select_layout_records(layout_records, split_records)

    metrics, errors_df = evaluate_records(records, parse_menu_lines)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"parser_metrics_{split_name}.json"
    errors_path = out_dir / f"parser_errors_{split_name}.csv"

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    errors_df.to_csv(errors_path, index=False)

    print("Parser module:", parser_module)
    print("Saved metrics:", metrics_path)
    print("Saved errors:", errors_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_path", type=str, default="data/processed/gold/menu_gold_all_batches_layout.jsonl")
    parser.add_argument("--split_reference_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--parser_module", type=str, required=True)
    args = parser.parse_args()

    main(
        layout_path=Path(args.layout_path),
        split_reference_path=Path(args.split_reference_path),
        out_dir=Path(args.out_dir),
        split_name=args.split_name,
        parser_module=args.parser_module,
    )

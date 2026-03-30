from pathlib import Path
import argparse
import json

import pandas as pd

from menu_structuring_baseline import parse_menu_lines


def normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def normalize_price(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.replace(",", ".").strip()

    try:
        return round(float(value), 2)
    except Exception:
        return None


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_parser_input(ocr_lines):
    result = []
    for line in ocr_lines:
        if isinstance(line, str):
            result.append({"text": line})
        else:
            result.append({"text": str(line)})
    return result


def build_gold_map(items):
    gold_map = {}

    for item in items:
        key = normalize_text(item.get("dish_name"))
        if key == "":
            continue

        if key not in gold_map:
            gold_map[key] = []

        gold_map[key].append(item)

    return gold_map


def evaluate_records(records):
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

        parser_input = make_parser_input(record["ocr_lines"])
        pred_items = parse_menu_lines(parser_input)
        gold_items = record["items"]

        gold_map = build_gold_map(gold_items)
        matched_gold = set()

        for pred in pred_items:
            pred_name = normalize_text(pred.get("dish_name"))
            if pred_name == "":
                item_fp += 1
                error_rows.append(
                    {
                        "menu_id": menu_id,
                        "page_id": page_id,
                        "error_type": "empty_pred_name",
                        "pred_name": None,
                        "gold_name": None,
                    }
                )
                continue

            gold_candidates = gold_map.get(pred_name, [])

            match_index = None
            for i, gold in enumerate(gold_candidates):
                gold_key = f"{pred_name}__{i}"
                if gold_key not in matched_gold:
                    match_index = i
                    matched_gold.add(gold_key)
                    break

            if match_index is None:
                item_fp += 1
                error_rows.append(
                    {
                        "menu_id": menu_id,
                        "page_id": page_id,
                        "error_type": "false_positive_item",
                        "pred_name": pred.get("dish_name"),
                        "gold_name": None,
                    }
                )
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
                    error_rows.append(
                        {
                            "menu_id": menu_id,
                            "page_id": page_id,
                            "error_type": "wrong_section",
                            "pred_name": pred.get("dish_name"),
                            "gold_name": gold_item.get("dish_name"),
                        }
                    )

            gold_price = normalize_price(gold_item.get("price_value"))
            pred_price = normalize_price(pred.get("price_value"))

            if gold_price is not None:
                price_total += 1
                if pred_price == gold_price:
                    price_ok += 1
                else:
                    error_rows.append(
                        {
                            "menu_id": menu_id,
                            "page_id": page_id,
                            "error_type": "wrong_price",
                            "pred_name": pred.get("dish_name"),
                            "gold_name": gold_item.get("dish_name"),
                        }
                    )

            gold_desc = normalize_text(gold_item.get("description"))
            pred_desc = normalize_text(pred.get("description"))

            if gold_desc != "":
                description_total += 1
                if gold_desc == pred_desc:
                    description_ok += 1
                else:
                    error_rows.append(
                        {
                            "menu_id": menu_id,
                            "page_id": page_id,
                            "error_type": "wrong_description",
                            "pred_name": pred.get("dish_name"),
                            "gold_name": gold_item.get("dish_name"),
                        }
                    )

        total_gold = 0
        for key, values in gold_map.items():
            total_gold += len(values)
            for i, gold in enumerate(values):
                gold_key = f"{key}__{i}"
                if gold_key not in matched_gold:
                    item_fn += 1
                    error_rows.append(
                        {
                            "menu_id": menu_id,
                            "page_id": page_id,
                            "error_type": "missed_gold_item",
                            "pred_name": None,
                            "gold_name": gold.get("dish_name"),
                        }
                    )

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

    errors_df = pd.DataFrame(error_rows)
    return metrics, errors_df


def main(input_path, out_dir, split_name):
    records = load_jsonl(input_path)
    metrics, errors_df = evaluate_records(records)

    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"parser_metrics_{split_name}.json"
    errors_path = out_dir / f"parser_errors_{split_name}.csv"

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    errors_df.to_csv(errors_path, index=False)

    print("Saved metrics:", metrics_path)
    print("Saved errors:", errors_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="reports/parser_baseline")
    parser.add_argument("--split_name", type=str, default="test")
    args = parser.parse_args()

    main(
        input_path=Path(args.input_path),
        out_dir=Path(args.out_dir),
        split_name=args.split_name,
    )
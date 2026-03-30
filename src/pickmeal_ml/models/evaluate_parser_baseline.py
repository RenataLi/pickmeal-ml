from pathlib import Path
import argparse
import json

import pandas as pd

from pickmeal_ml.models.menu_structuring_baseline import OCRLine, parse_menu_lines


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_lines(ocr_lines):
    result = []
    for line in ocr_lines:
        if isinstance(line, str):
            result.append(OCRLine(text=line))
        else:
            result.append(OCRLine(text=line.get("text", "")))
    return result


def safe_price(value):
    if value is None:
        return None
    return float(value)


def evaluate(records):
    item_tp = 0
    item_fp = 0
    item_fn = 0

    section_ok = 0
    section_total = 0

    description_ok = 0
    description_total = 0

    price_ok = 0
    price_total = 0
    price_abs_errors = []

    error_rows = []

    for row in records:
        menu_id = row["menu_id"]
        gold_items = row["items"]
        pred_items = parse_menu_lines(make_lines(row["ocr_lines"]))

        pred_map = {}
        for item in pred_items:
            key = normalize_text(item.dish_name)
            if key not in pred_map:
                pred_map[key] = []
            pred_map[key].append(item)

        matched_pred = 0

        for gold in gold_items:
            gold_name = normalize_text(gold.get("dish_name"))
            pred = None

            if gold_name in pred_map and len(pred_map[gold_name]) > 0:
                pred = pred_map[gold_name].pop(0)
                matched_pred += 1

            if pred is None:
                item_fn += 1
                error_rows.append(
                    {
                        "menu_id": menu_id,
                        "error_type": "missed_item",
                        "gold_name": gold.get("dish_name"),
                        "pred_name": None,
                    }
                )
                continue

            item_tp += 1

            gold_section = gold.get("section")
            if gold_section is not None:
                section_total += 1
                if normalize_text(gold_section) == normalize_text(pred.section):
                    section_ok += 1

            gold_desc = gold.get("description")
            if gold_desc is not None:
                description_total += 1
                if normalize_text(gold_desc) == normalize_text(pred.description):
                    description_ok += 1

            gold_price = gold.get("price_value")
            pred_price = pred.price_value

            if gold_price is not None:
                price_total += 1
                gold_price = safe_price(gold_price)

                if pred_price is not None:
                    diff = abs(gold_price - float(pred_price))
                    price_abs_errors.append(diff)

                    if diff < 0.01:
                        price_ok += 1

        item_fp += max(len(pred_items) - matched_pred, 0)

    precision = item_tp / (item_tp + item_fp) if (item_tp + item_fp) > 0 else 0.0
    recall = item_tp / (item_tp + item_fn) if (item_tp + item_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "n_menus": len(records),
        "item_precision": round(precision, 4),
        "item_recall": round(recall, 4),
        "item_f1": round(f1, 4),
        "section_accuracy": round(section_ok / section_total, 4) if section_total > 0 else None,
        "description_accuracy": round(description_ok / description_total, 4) if description_total > 0 else None,
        "price_accuracy": round(price_ok / price_total, 4) if price_total > 0 else None,
        "price_mae": round(sum(price_abs_errors) / len(price_abs_errors), 4) if price_abs_errors else None,
        "item_tp": item_tp,
        "item_fp": item_fp,
        "item_fn": item_fn,
    }

    return report, error_rows


def main(gold_path, report_path, errors_path):
    records = load_jsonl(gold_path)
    report, error_rows = evaluate(records)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(error_rows).to_csv(errors_path, index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved report to:", report_path)
    print("Saved errors to:", errors_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--report_path", type=str, default="reports/parser_baseline_report.json")
    parser.add_argument("--errors_path", type=str, default="reports/parser_baseline_errors.csv")
    args = parser.parse_args()

    main(
        gold_path=Path(args.gold_path),
        report_path=Path(args.report_path),
        errors_path=Path(args.errors_path),
    )
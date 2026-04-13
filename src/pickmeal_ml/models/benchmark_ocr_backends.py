from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

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


def load_page_map(annotation_batches_dir: Path):
    page_map = {}
    for pages_path in sorted(annotation_batches_dir.rglob("batch_*_pages.csv")):
        pages_df = pd.read_csv(pages_path)
        for _, row in pages_df.iterrows():
            key = (str(row["menu_id"]).strip(), str(row["page_id"]).strip())
            page_map[key] = str(row["image_path"]).strip()
    return page_map


def build_gold_map(items):
    gold_map = {}
    for item in items:
        key = normalize_text(item.get("dish_name"))
        if key:
            gold_map.setdefault(key, []).append(item)
    return gold_map


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

    for record in records:
        pred_items = parse_menu_lines(record["ocr_lines"])
        gold_items = record["items"]
        gold_map = build_gold_map(gold_items)
        matched_gold = set()

        for pred in pred_items:
            pred_name = normalize_text(pred.get("dish_name"))
            if pred_name == "":
                item_fp += 1
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
                continue

            item_tp += 1
            gold_item = gold_candidates[match_index]

            gold_section = normalize_text(gold_item.get("section"))
            pred_section = normalize_text(pred.get("section"))
            if gold_section != "":
                section_total += 1
                if gold_section == pred_section:
                    section_ok += 1

            gold_price = normalize_price(gold_item.get("price_value"))
            pred_price = normalize_price(pred.get("price_value"))
            if gold_price is not None:
                price_total += 1
                if pred_price == gold_price:
                    price_ok += 1

            gold_desc = normalize_text(gold_item.get("description"))
            pred_desc = normalize_text(pred.get("description"))
            if gold_desc != "":
                description_total += 1
                if gold_desc == pred_desc:
                    description_ok += 1

        for key, values in gold_map.items():
            for i, _ in enumerate(values):
                gold_key = f"{key}__{i}"
                if gold_key not in matched_gold:
                    item_fn += 1

    precision = item_tp / (item_tp + item_fp) if (item_tp + item_fp) > 0 else 0.0
    recall = item_tp / (item_tp + item_fn) if (item_tp + item_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_pages": len(records),
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


def main(input_path: Path, annotation_batches_dir: Path, out_dir: Path, parser_module: str, backends: list[str], langs: str, max_records: int | None):
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from service.app.services.ocr_service import run_ocr

    module = importlib.import_module(parser_module)
    parse_menu_lines = getattr(module, "parse_menu_lines")

    records = load_jsonl(input_path)
    if max_records is not None:
        records = records[:max_records]

    page_map = load_page_map(annotation_batches_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    sample_rows = []

    for backend in backends:
        backend_records = []
        for record in records:
            key = (str(record["menu_id"]).strip(), str(record.get("page_id", "p1")).strip())
            image_rel_path = page_map.get(key)
            if not image_rel_path:
                continue
            image_path = project_root / image_rel_path
            if not image_path.exists():
                continue

            ocr_result = run_ocr(image_path.read_bytes(), langs=langs, backend=backend)
            ocr_lines = [line.model_dump() for line in ocr_result.lines]
            pred_items = parse_menu_lines(ocr_lines)
            avg_conf = None
            if ocr_lines:
                scores = [row.get("ocr_confidence") for row in ocr_lines if row.get("ocr_confidence") is not None]
                if scores:
                    avg_conf = round(sum(scores) / len(scores), 4)

            backend_records.append(
                {
                    "menu_id": record["menu_id"],
                    "page_id": record.get("page_id", "p1"),
                    "ocr_lines": ocr_lines,
                    "items": record["items"],
                }
            )
            sample_rows.append(
                {
                    "backend": ocr_result.backend,
                    "menu_id": record["menu_id"],
                    "page_id": record.get("page_id", "p1"),
                    "n_ocr_lines": len(ocr_lines),
                    "n_pred_items": len(pred_items),
                    "avg_ocr_confidence": avg_conf,
                }
            )

        metrics = evaluate_records(backend_records, parse_menu_lines)
        backend_sample = [row for row in sample_rows if row["backend"] == backend]
        metrics["avg_ocr_lines"] = round(sum(row["n_ocr_lines"] for row in backend_sample) / len(backend_sample), 2) if backend_sample else 0.0
        conf_rows = [row["avg_ocr_confidence"] for row in backend_sample if row["avg_ocr_confidence"] is not None]
        metrics["avg_ocr_confidence"] = round(sum(conf_rows) / len(conf_rows), 4) if conf_rows else None
        metrics["backend"] = backend
        summary_rows.append(metrics)

    pd.DataFrame(summary_rows).to_csv(out_dir / "ocr_backend_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(out_dir / "ocr_backend_samples.csv", index=False)
    (out_dir / "ocr_backend_metrics.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/processed/gold/full_splits/test.jsonl")
    parser.add_argument("--annotation_batches_dir", type=str, default="data/processed/annotation_batches")
    parser.add_argument("--out_dir", type=str, default="reports/ocr_backend_benchmark")
    parser.add_argument("--parser_module", type=str, default="pickmeal_ml.models.menu_structuring_baseline_v2")
    parser.add_argument("--backends", nargs="+", default=["easyocr", "paddleocr"])
    parser.add_argument("--langs", type=str, default="en")
    parser.add_argument("--max_records", type=int, default=12)
    args = parser.parse_args()

    main(
        input_path=Path(args.input_path),
        annotation_batches_dir=Path(args.annotation_batches_dir),
        out_dir=Path(args.out_dir),
        parser_module=args.parser_module,
        backends=args.backends,
        langs=args.langs,
        max_records=args.max_records,
    )

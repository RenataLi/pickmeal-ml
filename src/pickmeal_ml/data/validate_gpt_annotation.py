from pathlib import Path
import argparse
import ast
import json

import pandas as pd


REQUIRED_COLUMNS = [
    "menu_id",
    "page_id",
    "item_id",
    "section",
    "dish_name",
    "price_text",
    "price_value",
    "currency",
    "description",
    "explicit_ingredients",
    "explicit_allergens",
    "source_line_ids",
    "confidence",
    "needs_review",
    "review_reason",
]

LIST_COLUMNS = [
    "explicit_ingredients",
    "explicit_allergens",
    "source_line_ids",
]


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def normalize_text(value):
    if isinstance(value, (list, dict)):
        return value

    if pd.isna(value):
        return None

    text = str(value).strip()
    if text == "":
        return None

    return text


def parse_list_value(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() != ""]

    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]

    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() != ""]

    text = str(value).strip()
    if text == "":
        return []

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = [x.strip() for x in text.split("|") if x.strip() != ""]

    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip() != ""]

    if parsed is None:
        return []

    parsed_text = str(parsed).strip()
    return [parsed_text] if parsed_text != "" else []


def normalize_dataframe(df):
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(normalize_text)

    for col in LIST_COLUMNS:
        if col not in out.columns:
            out[col] = [[] for _ in range(len(out))]
        out[col] = out[col].apply(parse_list_value)
        out[col] = out[col].apply(json.dumps)

    if "needs_review" in out.columns:
        out["needs_review"] = (
            out["needs_review"]
            .fillna(False)
            .replace(
                {
                    "true": True,
                    "false": False,
                    "True": True,
                    "False": False,
                    "1": True,
                    "0": False,
                }
            )
            .astype(bool)
        )
    else:
        out["needs_review"] = False

    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    else:
        out["confidence"] = None

    if "price_value" in out.columns:
        out["price_value"] = pd.to_numeric(out["price_value"], errors="coerce")
    else:
        out["price_value"] = None

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    return out[REQUIRED_COLUMNS].copy()


def load_annotation_frame(in_path):
    suffix = in_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(in_path)

    if suffix == ".json":
        payload = json.loads(in_path.read_text())
        if isinstance(payload, dict):
            if "items" in payload and isinstance(payload["items"], list):
                payload = payload["items"]
            else:
                payload = [payload]
        return pd.DataFrame(payload)

    if suffix == ".jsonl":
        rows = []
        for line in in_path.read_text().splitlines():
            line = line.strip()
            if line == "":
                continue
            rows.append(json.loads(line))
        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported file format: {suffix}")


def build_validation_issues(df):
    issue_rows = []

    for idx, row in df.iterrows():
        row_id = {
            "row_index": int(idx),
            "menu_id": row.get("menu_id"),
            "page_id": row.get("page_id"),
            "item_id": row.get("item_id"),
        }

        dish_name = normalize_text(row.get("dish_name"))
        price_text = normalize_text(row.get("price_text"))
        review_reason = normalize_text(row.get("review_reason"))
        confidence = row.get("confidence")
        needs_review = bool(row.get("needs_review"))

        if dish_name is None:
            issue_rows.append({**row_id, "issue": "missing dish_name"})

        if confidence is not None and not pd.isna(confidence):
            if float(confidence) < 0 or float(confidence) > 1:
                issue_rows.append({**row_id, "issue": "confidence out of range"})

        if needs_review and review_reason is None:
            issue_rows.append({**row_id, "issue": "needs_review without review_reason"})

        if (price_text is None) and (not pd.isna(row.get("price_value"))):
            issue_rows.append({**row_id, "issue": "price_value without price_text"})

        if (price_text is not None) and pd.isna(row.get("price_value")) and not needs_review:
            issue_rows.append({**row_id, "issue": "price_text without price_value and no review flag"})

        if dish_name is not None and len(dish_name) <= 2:
            issue_rows.append({**row_id, "issue": "dish_name too short"})

    return pd.DataFrame(issue_rows)


def main(in_path, out_dir):
    df = load_annotation_frame(in_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_df = normalize_dataframe(df)
    issues_df = build_validation_issues(normalized_df)

    normalized_path = out_dir / f"{in_path.stem}_normalized.csv"
    issues_path = out_dir / f"{in_path.stem}_issues.csv"
    stats_path = out_dir / f"{in_path.stem}_validation_stats.json"

    normalized_df.to_csv(normalized_path, index=False)
    issues_df.to_csv(issues_path, index=False)

    stats = {
        "input_path": str(in_path),
        "n_rows": int(len(df)),
        "n_rows_normalized": int(len(normalized_df)),
        "n_issues": int(len(issues_df)),
        "missing_required_columns": missing_cols,
    }

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print("Saved:")
    print(normalized_path)
    print(issues_path)
    print(stats_path)
    print()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed/annotation_validation")
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        in_path=project_root / args.in_path,
        out_dir=project_root / args.out_dir,
    )

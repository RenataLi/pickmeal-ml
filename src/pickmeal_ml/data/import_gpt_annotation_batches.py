from pathlib import Path
import argparse
import json
import re

import pandas as pd
from PIL import Image


LIST_COLUMNS = [
    "explicit_ingredients",
    "explicit_allergens",
    "source_line_ids",
]


BATCH_CONFIGS = {
    "batch_004": {
        "mode": "rank_ranges",
        "files": [
            {"name": "output_menu.json", "rank_start": 1, "rank_end": 12},
            {"name": "out_3.json", "rank_start": 13, "rank_end": 24},
        ],
    },
    "batch_005": {
        "mode": "rank_ranges",
        "files": [
            {"name": "batch_005_001.json", "rank_start": 1, "rank_end": 12},
            {"name": "batch_006_001.json", "rank_start": 13, "rank_end": 24},
        ],
    },
    "batch_006": {
        "mode": "rank_ranges",
        "files": [
            {"name": "current_batch_006_001_output.json", "rank_start": 1, "rank_end": 12},
            {"name": "batch_006_002_current.json", "rank_start": 13, "rank_end": 24},
        ],
    },
    "batch_007": {
        "mode": "image_name",
        "files": [
            {"name": "batch_007_001.json"},
            {"name": "batch_007_002.json"},
        ],
    },
}


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def load_json_df(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            payload = payload["items"]
        else:
            payload = [payload]
    return pd.DataFrame(payload)


def normalize_text(value):
    if isinstance(value, (list, dict)):
        return value
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text != "" else None


def parse_list_value(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip() != ""]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    if value is None:
        return []
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text == "":
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip() != ""]
    except Exception:
        pass

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    return bool(value)


def normalize_float(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.replace(",", ".").strip()
    try:
        return float(value)
    except Exception:
        return None


def extract_page_number(page_id):
    text = str(page_id or "").strip().lower()
    match = re.match(r"p(\d+)$", text)
    if match:
        return int(match.group(1))
    return None


def prepare_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(normalize_text)

    for col in LIST_COLUMNS:
        if col not in out.columns:
            out[col] = [[] for _ in range(len(out))]
        out[col] = out[col].apply(parse_list_value)

    if "confidence" not in out.columns:
        out["confidence"] = None
    out["confidence"] = out["confidence"].apply(normalize_float)

    if "price_value" not in out.columns:
        out["price_value"] = None
    out["price_value"] = out["price_value"].apply(normalize_float)

    if "needs_review" not in out.columns:
        out["needs_review"] = False
    out["needs_review"] = out["needs_review"].apply(normalize_bool)

    if "review_reason" not in out.columns:
        out["review_reason"] = None

    if "section" not in out.columns:
        out["section"] = None
    if "dish_name" not in out.columns:
        out["dish_name"] = None
    if "price_text" not in out.columns:
        out["price_text"] = None
    if "currency" not in out.columns:
        out["currency"] = None
    if "description" not in out.columns:
        out["description"] = None

    out["source_json"] = source_name
    out["local_page_num"] = out["page_id"].apply(extract_page_number)
    out["local_page_id"] = out["page_id"].astype(str)
    return out


def regenerate_item_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.reset_index(drop=True)
    out["_row_order"] = range(len(out))
    out["item_id"] = None

    for (menu_id, page_id), group in out.groupby(["menu_id", "page_id"], sort=False):
        idx = group.sort_values("_row_order").index.tolist()
        for order, row_idx in enumerate(idx, start=1):
            out.at[row_idx, "item_id"] = f"{menu_id}_item_{order:03d}"

    out = out.drop(columns=["_row_order"])
    return out


def write_batch_outputs(batch_dir: Path, pages_df: pd.DataFrame, gold_df: pd.DataFrame, stats: dict):
    batch_name = batch_dir.name
    gold_path = batch_dir / f"menu_gold_{batch_name}_final_reviewed.csv"
    stats_path = batch_dir / f"{batch_name}_import_stats.json"

    if "qa_note" not in gold_df.columns:
        gold_df["qa_note"] = None

    gold_df.to_csv(gold_path, index=False)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", gold_path)
    print("Saved:", stats_path)


def build_rank_range_batch(project_root: Path, batch_dir: Path, config: dict):
    batch_name = batch_dir.name
    pages_path = batch_dir / f"{batch_name}_pages.csv"
    pages_df = pd.read_csv(pages_path).copy().sort_values("selection_rank").reset_index(drop=True)

    frames = []
    stats_files = []

    for file_cfg in config["files"]:
        path = batch_dir / file_cfg["name"]
        raw_df = prepare_rows(load_json_df(path), source_name=path.name)
        subset = pages_df[
            (pages_df["selection_rank"] >= file_cfg["rank_start"]) &
            (pages_df["selection_rank"] <= file_cfg["rank_end"])
        ].copy().sort_values("selection_rank")

        mapping = {
            local_idx: {
                "menu_id": row["menu_id"],
                "page_id": row["page_id"],
                "image_file": row["image_file"],
                "image_path": row["image_path"],
                "selection_rank": int(row["selection_rank"]),
            }
            for local_idx, (_, row) in enumerate(subset.iterrows(), start=1)
        }

        raw_df = raw_df[raw_df["local_page_num"].notna()].copy()
        raw_df["local_page_num"] = raw_df["local_page_num"].astype(int)
        raw_df = raw_df[raw_df["local_page_num"].isin(mapping.keys())].copy()

        raw_df["menu_id"] = raw_df["local_page_num"].map(lambda x: mapping[x]["menu_id"])
        raw_df["page_id"] = raw_df["local_page_num"].map(lambda x: mapping[x]["page_id"])
        raw_df["image_file"] = raw_df["local_page_num"].map(lambda x: mapping[x]["image_file"])
        raw_df["image_path"] = raw_df["local_page_num"].map(lambda x: mapping[x]["image_path"])
        raw_df["selection_rank"] = raw_df["local_page_num"].map(lambda x: mapping[x]["selection_rank"])
        raw_df["qa_note"] = (
            "imported_from="
            + raw_df["source_json"].astype(str)
            + "; local_page_id="
            + raw_df["local_page_id"].astype(str)
        )

        frames.append(raw_df)
        stats_files.append(
            {
                "source_json": path.name,
                "rank_start": file_cfg["rank_start"],
                "rank_end": file_cfg["rank_end"],
                "n_rows": int(len(raw_df)),
                "n_pages_present": int(raw_df["menu_id"].nunique()),
            }
        )

    gold_df = pd.concat(frames, ignore_index=True)
    gold_df = regenerate_item_ids(gold_df)

    keep_cols = [
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
        "qa_note",
    ]

    for col in LIST_COLUMNS:
        gold_df[col] = gold_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

    gold_df = gold_df[keep_cols].copy()

    stats = {
        "batch_name": batch_name,
        "mode": "rank_ranges",
        "expected_pages": int(len(pages_df)),
        "actual_pages_with_items": int(gold_df["menu_id"].nunique()),
        "n_rows": int(len(gold_df)),
        "source_files": stats_files,
    }

    write_batch_outputs(batch_dir=batch_dir, pages_df=pages_df, gold_df=gold_df, stats=stats)


def get_image_size(image_path: Path):
    with Image.open(image_path) as img:
        width, height = img.size
    return float(width), float(height)


def backup_if_needed(path: Path, suffix: str):
    backup_path = path.with_name(path.stem + suffix + path.suffix)
    if path.exists() and not backup_path.exists():
        path.replace(backup_path)
        return backup_path
    return backup_path if backup_path.exists() else None


def build_image_name_batch(project_root: Path, batch_dir: Path, config: dict):
    batch_name = batch_dir.name
    images_dir = batch_dir / "images"

    frames = []
    image_order = []
    seen_images = set()
    stats_files = []

    for file_cfg in config["files"]:
        path = batch_dir / file_cfg["name"]
        raw_df = prepare_rows(load_json_df(path), source_name=path.name)
        raw_df = raw_df[raw_df["image_name"].notna()].copy()

        file_image_order = []
        for image_name in raw_df["image_name"].astype(str):
            if image_name not in seen_images:
                seen_images.add(image_name)
                image_order.append(image_name)
                file_image_order.append(image_name)

        frames.append(raw_df)
        stats_files.append(
            {
                "source_json": path.name,
                "n_rows": int(len(raw_df)),
                "n_unique_images": int(raw_df["image_name"].nunique()),
            }
        )

    pages_rows = []
    image_mapping = {}

    for idx, image_name in enumerate(image_order, start=1):
        menu_id = f"{batch_name}_{idx:03d}"
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image referenced in JSON was not found: {image_path}")

        img_w, img_h = get_image_size(image_path)
        rel_image_path = str(image_path.relative_to(project_root))

        row = {
            "menu_id": menu_id,
            "page_id": "p1",
            "image_file": image_name,
            "image_path": rel_image_path,
            "source_image_path": rel_image_path,
            "img_w": img_w,
            "img_h": img_h,
            "selection_rank": idx,
            "selection_reason": "custom image_name import from GPT batch",
        }
        pages_rows.append(row)
        image_mapping[image_name] = row

    pages_df = pd.DataFrame(pages_rows)

    existing_pages_path = batch_dir / f"{batch_name}_pages.csv"
    existing_ocr_path = batch_dir / f"{batch_name}_ocr.csv"
    backup_if_needed(existing_pages_path, "_generated_backup")
    backup_if_needed(existing_ocr_path, "_generated_backup")
    pages_df.to_csv(existing_pages_path, index=False)

    gold_df = pd.concat(frames, ignore_index=True)
    gold_df["menu_id"] = gold_df["image_name"].astype(str).map(lambda x: image_mapping[x]["menu_id"])
    gold_df["page_id"] = "p1"
    gold_df["qa_note"] = "imported_from=" + gold_df["source_json"].astype(str) + "; image_name=" + gold_df["image_name"].astype(str)
    gold_df = regenerate_item_ids(gold_df)

    for col in LIST_COLUMNS:
        gold_df[col] = gold_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))

    keep_cols = [
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
        "qa_note",
    ]
    gold_df = gold_df[keep_cols].copy()

    unused_images = sorted(
        p.name
        for p in images_dir.glob("*")
        if p.is_file() and not p.name.startswith(".") and p.name not in image_mapping
    )
    unused_path = batch_dir / f"{batch_name}_unused_images.csv"
    pd.DataFrame({"image_file": unused_images}).to_csv(unused_path, index=False)

    stats = {
        "batch_name": batch_name,
        "mode": "image_name",
        "actual_pages_with_items": int(gold_df["menu_id"].nunique()),
        "n_rows": int(len(gold_df)),
        "source_files": stats_files,
        "n_unused_images": len(unused_images),
    }

    write_batch_outputs(batch_dir=batch_dir, pages_df=pages_df, gold_df=gold_df, stats=stats)
    print("Saved:", existing_pages_path)
    print("Saved:", unused_path)


def main(batch_names):
    project_root = resolve_project_root()
    annotation_root = project_root / "data" / "processed" / "annotation_batches"

    for batch_name in batch_names:
        if batch_name not in BATCH_CONFIGS:
            raise ValueError(f"No config found for {batch_name}")

        batch_dir = annotation_root / batch_name
        config = BATCH_CONFIGS[batch_name]

        print()
        print("Processing:", batch_name)

        if config["mode"] == "rank_ranges":
            build_rank_range_batch(project_root, batch_dir, config)
        elif config["mode"] == "image_name":
            build_image_name_batch(project_root, batch_dir, config)
        else:
            raise ValueError(f"Unsupported mode: {config['mode']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_names",
        nargs="+",
        default=["batch_004", "batch_005", "batch_006", "batch_007"],
    )
    args = parser.parse_args()
    main(args.batch_names)

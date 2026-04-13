from pathlib import Path
import argparse
import json

import pandas as pd


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


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


def main(batch_dir, in_path):
    batch_dir = Path(batch_dir)
    pages_files = sorted(batch_dir.glob("*_pages.csv"))

    if len(pages_files) != 1:
        raise FileNotFoundError(f"Expected exactly one *_pages.csv in {batch_dir}, found {len(pages_files)}")

    pages_df = pd.read_csv(pages_files[0]).copy()
    ann_df = load_annotation_frame(in_path).copy()

    expected = pages_df[["menu_id", "page_id", "image_file", "source_image_path"]].drop_duplicates().copy()
    actual = ann_df[["menu_id", "page_id"]].drop_duplicates().copy()

    expected_pairs = set(map(tuple, expected[["menu_id", "page_id"]].itertuples(index=False, name=None)))
    actual_pairs = set(map(tuple, actual[["menu_id", "page_id"]].itertuples(index=False, name=None)))

    missing_pairs = sorted(expected_pairs - actual_pairs)
    extra_pairs = sorted(actual_pairs - expected_pairs)

    expected_df = pd.DataFrame(missing_pairs, columns=["menu_id", "page_id"]) if missing_pairs else pd.DataFrame(columns=["menu_id", "page_id"])
    if len(expected_df) > 0:
        expected_df = expected_df.merge(
            expected,
            on=["menu_id", "page_id"],
            how="left",
        )

    extra_df = pd.DataFrame(extra_pairs, columns=["menu_id", "page_id"]) if extra_pairs else pd.DataFrame(columns=["menu_id", "page_id"])

    stats = {
        "batch_dir": str(batch_dir),
        "annotation_path": str(in_path),
        "expected_pages": int(len(expected)),
        "actual_unique_pages": int(len(actual)),
        "missing_pages": int(len(expected_df)),
        "extra_pages": int(len(extra_df)),
        "coverage_ratio": round(float(len(expected_pairs & actual_pairs) / max(len(expected_pairs), 1)), 4),
    }

    out_prefix = in_path.stem
    out_dir = batch_dir / "coverage_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir / f"{out_prefix}_coverage_stats.json"
    missing_path = out_dir / f"{out_prefix}_missing_pages.csv"
    extra_path = out_dir / f"{out_prefix}_extra_pages.csv"

    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    expected_df.to_csv(missing_path, index=False)
    extra_df.to_csv(extra_path, index=False)

    print("Saved:")
    print(stats_path)
    print(missing_path)
    print(extra_path)
    print()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True)
    parser.add_argument("--in_path", type=str, required=True)
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        batch_dir=project_root / args.batch_dir,
        in_path=project_root / args.in_path,
    )

from pathlib import Path
import argparse
import pandas as pd


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def find_gold_files(search_dir):
    patterns = [
        "menu_gold_batch_*_final_reviewed.csv",
        "menu_gold_batch_*_final.csv",
        "menu_gold_batch_*_fixed.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(search_dir.rglob(pattern))

    files = sorted(set(files))
    return files


def main(search_dir, out_path):
    files = find_gold_files(search_dir)

    if len(files) == 0:
        raise FileNotFoundError("No gold CSV files were found.")

    frames = []
    source_rows = []

    for path in files:
        df = pd.read_csv(path).copy()
        df["source_file"] = path.name
        frames.append(df)

        source_rows.append(
            {
                "source_file": path.name,
                "n_rows": len(df),
            }
        )

    merged = pd.concat(frames, ignore_index=True)

    key_cols = [col for col in ["menu_id", "page_id", "item_id"] if col in merged.columns]

    if len(key_cols) > 0:
        before = len(merged)
        merged = merged.drop_duplicates(subset=key_cols, keep="first").copy()
        after = len(merged)
    else:
        before = len(merged)
        after = len(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    stats_path = out_path.parent / "merge_gold_stats.csv"
    pd.DataFrame(source_rows).to_csv(stats_path, index=False)

    print("Merged file:", out_path)
    print("Stats file:", stats_path)
    print("Files used:", len(files))
    print("Rows before dedup:", before)
    print("Rows after dedup:", after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_dir",
        type=str,
        default="data/processed",
        help="Directory where batch gold CSV files are stored",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/processed/gold/menu_gold_all_batches.csv",
        help="Path for the merged CSV",
    )
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        search_dir=project_root / args.search_dir,
        out_path=project_root / args.out_path,
    )

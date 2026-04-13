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


def main(batch_dir, wave_size):
    batch_dir = Path(batch_dir)
    pages_files = sorted(batch_dir.glob("*_pages.csv"))

    if len(pages_files) != 1:
        raise FileNotFoundError(f"Expected exactly one *_pages.csv in {batch_dir}, found {len(pages_files)}")

    pages_path = pages_files[0]
    df = pd.read_csv(pages_path).copy()

    df = df.sort_values(["selection_rank", "menu_id", "page_id", "image_file"]).reset_index(drop=True)
    df["wave_index"] = (df.index // wave_size) + 1
    df["wave_id"] = df["wave_index"].apply(lambda x: f"wave_{int(x):02d}")
    df["gpt_note"] = (
        "Send this image with its menu_id/page_id exactly as listed. "
        "Treat each image as a separate menu page."
    )

    out_cols = [
        "wave_id",
        "selection_rank",
        "menu_id",
        "page_id",
        "image_file",
        "image_path",
        "source_image_path",
        "layout_bucket",
        "selection_reason",
        "gpt_note",
    ]

    out_path = batch_dir / f"{batch_dir.name}_wave_plan.csv"
    df[out_cols].to_csv(out_path, index=False)

    summary = (
        df.groupby("wave_id")
        .agg(
            n_images=("image_file", "size"),
            first_rank=("selection_rank", "min"),
            last_rank=("selection_rank", "max"),
        )
        .reset_index()
    )
    summary_path = batch_dir / f"{batch_dir.name}_wave_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Saved:")
    print(out_path)
    print(summary_path)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", type=str, required=True)
    parser.add_argument("--wave_size", type=int, default=6)
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        batch_dir=project_root / args.batch_dir,
        wave_size=args.wave_size,
    )

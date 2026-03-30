from pathlib import Path
import argparse

import pandas as pd


AUTO_ACCEPT_REASONS = {
    "currency unclear",
}


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


def clean_dataframe(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(normalize_text)

    if "needs_review" in df.columns:
        df["needs_review"] = df["needs_review"].fillna(False).astype(bool)

    return df


def should_auto_accept(reason):
    if reason is None:
        return False

    parts = [x.strip() for x in str(reason).split(";")]
    parts = [x for x in parts if x]

    if len(parts) == 0:
        return False

    return all(part in AUTO_ACCEPT_REASONS for part in parts)


def main(gold_path, review_path, out_dir):
    gold_df = pd.read_csv(gold_path)
    review_df = pd.read_csv(review_path)

    gold_df = clean_dataframe(gold_df)
    review_df = clean_dataframe(review_df)

    gold_df["qa_note"] = None

    auto_accept_mask = gold_df["review_reason"].apply(should_auto_accept)

    gold_df.loc[auto_accept_mask, "qa_note"] = gold_df.loc[auto_accept_mask, "review_reason"]
    gold_df.loc[auto_accept_mask, "needs_review"] = False
    gold_df.loc[auto_accept_mask, "review_reason"] = None

    still_review_df = gold_df[gold_df["needs_review"] == True].copy()
    final_df = gold_df[gold_df["needs_review"] == False].copy()

    reason_stats = (
        review_df["review_reason"]
        .fillna("no_reason")
        .value_counts()
        .rename_axis("review_reason")
        .reset_index(name="count")
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = out_dir / "menu_gold_batch_001_final.csv"
    review_path_out = out_dir / "menu_manual_review_priority.csv"
    stats_path = out_dir / "menu_review_reason_stats.csv"

    final_df.to_csv(final_path, index=False)
    still_review_df.to_csv(review_path_out, index=False)
    reason_stats.to_csv(stats_path, index=False)

    print("Saved:")
    print(final_path)
    print(review_path_out)
    print(stats_path)
    print()

    print("Rows in original gold:", len(gold_df))
    print("Rows auto-accepted:", int(auto_accept_mask.sum()))
    print("Rows still needing manual review:", len(still_review_df))
    print("Rows in final gold:", len(final_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--review_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed/annotation_final")
    args = parser.parse_args()

    main(
        gold_path=Path(args.gold_path),
        review_path=Path(args.review_path),
        out_dir=Path(args.out_dir),
    )
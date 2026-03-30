from pathlib import Path
import argparse
import json

from sklearn.model_selection import GroupShuffleSplit


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(input_path, out_dir, test_size, valid_size, random_state):
    rows = load_jsonl(input_path)

    if len(rows) == 0:
        raise ValueError("Input JSONL is empty.")

    menu_ids = [row["menu_id"] for row in rows]
    indexes = list(range(len(rows)))

    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_valid_idx, test_idx = next(split1.split(indexes, groups=menu_ids))

    train_valid_rows = [rows[i] for i in train_valid_idx]
    test_rows = [rows[i] for i in test_idx]

    train_valid_groups = [row["menu_id"] for row in train_valid_rows]
    train_valid_indexes = list(range(len(train_valid_rows)))

    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=valid_size,
        random_state=random_state,
    )

    train_idx, valid_idx = next(split2.split(train_valid_indexes, groups=train_valid_groups))

    train_rows = [train_valid_rows[i] for i in train_idx]
    valid_rows = [train_valid_rows[i] for i in valid_idx]

    save_jsonl(out_dir / "train.jsonl", train_rows)
    save_jsonl(out_dir / "valid.jsonl", valid_rows)
    save_jsonl(out_dir / "test.jsonl", test_rows)

    stats = {
        "train_menus": len(train_rows),
        "valid_menus": len(valid_rows),
        "test_menus": len(test_rows),
        "test_size": test_size,
        "valid_size": valid_size,
        "random_state": random_state,
    }

    stats_path = out_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:")
    print(out_dir / "train.jsonl", len(train_rows))
    print(out_dir / "valid.jsonl", len(valid_rows))
    print(out_dir / "test.jsonl", len(test_rows))
    print("Saved:", stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/gold/menu_gold_all_batches.jsonl",
        help="Input JSONL with all menus",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/gold/full_splits",
        help="Output directory for train/valid/test JSONL",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    project_root = resolve_project_root()

    main(
        input_path=project_root / args.input_path,
        out_dir=project_root / args.out_dir,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.random_state,
    )

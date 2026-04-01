from pathlib import Path
import argparse
import json
import pandas as pd


def load_json(path, label):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["label"] = label
    return data


def main(v1_valid, v1_test, v2_valid, v2_test, out_csv):
    rows = [
        load_json(v1_valid, "Parser v1 valid"),
        load_json(v1_test, "Parser v1 test"),
        load_json(v2_valid, "Parser v2 valid"),
        load_json(v2_test, "Parser v2 test"),
    ]
    df = pd.DataFrame(rows)
    columns = [
        "label", "n_menus", "item_precision", "item_recall", "item_f1",
        "section_accuracy", "price_accuracy", "description_exact_match",
        "item_tp", "item_fp", "item_fn",
    ]
    df = df[columns]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print("Saved:", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1_valid", required=True)
    parser.add_argument("--v1_test", required=True)
    parser.add_argument("--v2_valid", required=True)
    parser.add_argument("--v2_test", required=True)
    parser.add_argument("--out_csv", default="reports/parser_comparison/parser_v1_vs_v2.csv")
    args = parser.parse_args()
    main(args.v1_valid, args.v1_test, args.v2_valid, args.v2_test, args.out_csv)

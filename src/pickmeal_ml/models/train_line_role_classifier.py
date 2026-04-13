from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC


NUMERIC_COLS = [
    "word_count",
    "char_count",
    "digit_count",
    "uppercase_ratio",
    "title_ratio",
    "comma_count",
    "has_price_token",
    "price_only_flag",
    "ocr_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_center_x",
    "bbox_center_y",
    "line_order",
]


def resolve_project_root():
    root = Path.cwd().resolve()
    if (root / "data").exists():
        return root
    if (root.parent / "data").exists():
        return root.parent
    raise FileNotFoundError("Project root with data/ was not found.")


def make_model(model_type: str):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1500, class_weight="balanced", n_jobs=None)
    if model_type == "linear_svc":
        return LinearSVC(class_weight="balanced")
    if model_type == "sgd_log_loss":
        return SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42)
    if model_type == "sgd_modified_huber":
        return SGDClassifier(loss="modified_huber", class_weight="balanced", random_state=42)
    raise ValueError(f"Unsupported model_type: {model_type}")


def make_pipeline(model_type: str):
    text_word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    text_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, lowercase=True)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", MaxAbsScaler()),
        ]
    )

    features = ColumnTransformer(
        transformers=[
            ("text_word", text_word, "text"),
            ("text_char", text_char, "text"),
            ("num", numeric_pipe, NUMERIC_COLS),
        ]
    )

    model = make_model(model_type)
    return Pipeline(steps=[("features", features), ("model", model)])


def split_groups(df: pd.DataFrame, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx = list(range(len(df)))
    train_idx, test_idx = next(splitter.split(idx, groups=df["menu_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def main(input_path: Path, out_dir: Path, test_size: float, random_state: int, model_type: str):
    df = pd.read_csv(input_path)
    if len(df) == 0:
        raise ValueError("Input dataset is empty.")

    train_df, test_df = split_groups(df, test_size=test_size, random_state=random_state)

    X_train = train_df[["text"] + NUMERIC_COLS]
    y_train = train_df["label"]
    X_test = test_df[["text"] + NUMERIC_COLS]
    y_test = test_df["label"]

    pipe = make_pipeline(model_type=model_type)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_menus": int(train_df["menu_id"].nunique()),
        "n_test_menus": int(test_df["menu_id"].nunique()),
        "model_type": model_type,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, pred, average="weighted")), 4),
    }

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=sorted(df["label"].unique()))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "line_role_logreg.joblib"
    metrics_path = out_dir / "line_role_metrics.json"
    report_path = out_dir / "line_role_classification_report.json"
    cm_path = out_dir / "line_role_confusion_matrix.csv"
    preds_path = out_dir / "line_role_test_predictions.csv"

    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = sorted(df["label"].unique())
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(cm_path)

    pred_df = test_df.copy()
    pred_df["pred_label"] = pred
    pred_df.to_csv(preds_path, index=False)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved report:", report_path)
    print("Saved confusion matrix:", cm_path)
    print("Saved test predictions:", preds_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/processed/gold/line_role_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="reports/line_role_baseline")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_type", type=str, default="logreg")
    args = parser.parse_args()

    project_root = resolve_project_root()
    main(
        input_path=project_root / args.input_path,
        out_dir=project_root / args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        model_type=args.model_type,
    )

from pathlib import Path
import argparse
import json

from ultralytics import YOLO


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def main(data_yaml, model_name, epochs, imgsz, batch, device, run_name):
    project_root = resolve_project_root()

    runs_root = project_root / "reports" / "yolo_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    print("Start training...")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_root),
        name=run_name,
        plots=True,
    )

    best_path = runs_root / run_name / "weights" / "best.pt"

    if not best_path.exists():
        raise FileNotFoundError(f"Best model was not found: {best_path}")

    print("Run validation on test split...")
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        device=device,
    )

    if hasattr(metrics, "results_dict"):
        metrics_dict = metrics.results_dict
    else:
        metrics_dict = {}

    metrics_path = runs_root / run_name / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Best model:", best_path)
    print("Metrics file:", metrics_path)
    print("Metrics:")
    print(json.dumps(metrics_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    main(
        data_yaml=Path(args.data_yaml),
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        run_name=args.run_name,
    )
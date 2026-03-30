import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def get_project_root(start_path=None):
    if start_path is None:
        start_path = Path.cwd()

    start_path = start_path.resolve()

    if (start_path / "data").exists():
        return start_path

    if (start_path.parent / "data").exists():
        return start_path.parent

    raise FileNotFoundError("Could not find project root with data folder.")


def iter_images(root_path):
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_class_names(data_yaml_path):
    text = data_yaml_path.read_text(encoding="utf-8", errors="ignore")
    config = yaml.safe_load(text)
    names = config.get("names", [])

    if isinstance(names, dict):
        return [names[key] for key in sorted(names)]

    return list(names)


def simple_dhash(image_path, hash_size=8):
    with Image.open(image_path) as image:
        gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        arr = np.asarray(gray, dtype=np.uint8)

    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if x else "0" for x in diff.flatten())
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def parse_label_file(label_path, class_names, split_name, image_rel_path):
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    rows = []

    if text == "":
        return rows

    for line in text.splitlines():
        parts = line.strip().split()

        if len(parts) == 5:
            class_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])
            label_source = "yolo"

        elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            xs = coords[0::2]
            ys = coords[1::2]
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            bw = xmax - xmin
            bh = ymax - ymin
            label_source = "polygon_to_bbox"

        else:
            raise ValueError(f"Bad label line in {label_path}: {line}")

        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = str(class_id)

        rows.append({
            "source": "roboflow",
            "split": split_name,
            "image_path": image_rel_path,
            "class_id": class_id,
            "class_name": class_name,
            "xc": xc,
            "yc": yc,
            "bw": bw,
            "bh": bh,
            "annotation_source": label_source,
            "bbox_area_frac": bw * bh,
        })

    return rows


def scan_roboflow(project_root):
    rf_root = project_root / "data" / "raw" / "roboflow_menu_text_box_v3" / "Menu Text Box"
    data_yaml_path = rf_root / "data.yaml"
    class_names = load_class_names(data_yaml_path)

    image_rows = []
    box_rows = []

    for split_name in ["train", "valid", "test"]:
        image_dir = rf_root / split_name / "images"
        label_dir = rf_root / split_name / "labels"
        label_map = {path.stem: path for path in label_dir.rglob("*.txt")}

        for image_path in sorted(iter_images(image_dir)):
            image_rel = str(image_path.relative_to(project_root))
            label_path = label_map.get(image_path.stem)
            label_rel = None
            label_bytes = None
            ok = 1
            error_type = ""
            img_w = None
            img_h = None
            image_hash = None

            if label_path is not None and label_path.exists():
                label_rel = str(label_path.relative_to(project_root))
                label_bytes = label_path.stat().st_size

            try:
                with Image.open(image_path) as image:
                    img_w, img_h = image.size
                image_hash = simple_dhash(image_path)
            except Exception as e:
                ok = 0
                error_type = type(e).__name__

            is_tiny = 0
            if img_w is not None and img_h is not None:
                if min(img_w, img_h) < 150:
                    is_tiny = 1

            image_rows.append({
                "source": "roboflow",
                "split": split_name,
                "image_path": image_rel,
                "label_path": label_rel,
                "img_w": img_w,
                "img_h": img_h,
                "image_bytes": image_path.stat().st_size,
                "label_bytes": label_bytes,
                "ok": ok,
                "error_type": error_type,
                "is_tiny": is_tiny,
                "dhash": image_hash,
            })

            if ok == 1 and label_path is not None and label_path.exists():
                box_rows.extend(parse_label_file(label_path, class_names, split_name, image_rel))

    return image_rows, box_rows, class_names


def scan_kaggle(project_root):
    kg_root = project_root / "data" / "raw" / "kaggle_indian_menu_cards"
    rows = []

    for image_path in sorted(iter_images(kg_root)):
        image_rel = str(image_path.relative_to(project_root))
        ok = 1
        error_type = ""
        img_w = None
        img_h = None
        image_hash = None

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                img_w, img_h = image.size
            image_hash = simple_dhash(image_path)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            ok = 0
            error_type = type(e).__name__

        is_tiny = 0
        if img_w is not None and img_h is not None:
            if min(img_w, img_h) < 150:
                is_tiny = 1

        rows.append({
            "source": "kaggle",
            "split": "all",
            "image_path": image_rel,
            "label_path": None,
            "img_w": img_w,
            "img_h": img_h,
            "image_bytes": image_path.stat().st_size,
            "label_bytes": None,
            "ok": ok,
            "error_type": error_type,
            "is_tiny": is_tiny,
            "dhash": image_hash,
        })

    return rows


def save_csv(path, rows):
    rows = list(rows)
    if len(rows) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(image_rows, box_rows):
    summary = {}
    sources = sorted(set(row["source"] for row in image_rows))

    for source in sources:
        source_images = [row for row in image_rows if row["source"] == source]
        source_boxes = [row for row in box_rows if row["source"] == source]
        valid_images = [row for row in source_images if row["ok"] == 1]

        source_summary = {
            "n_images": len(source_images),
            "n_valid_images": len(valid_images),
            "n_corrupted_images": sum(1 for row in source_images if row["ok"] == 0),
            "n_tiny_images": sum(row["is_tiny"] for row in source_images),
            "n_boxes": len(source_boxes),
        }

        widths = [row["img_w"] for row in valid_images if row["img_w"] is not None]
        heights = [row["img_h"] for row in valid_images if row["img_h"] is not None]

        if len(widths) > 0:
            source_summary["img_w_mean"] = float(np.mean(widths))
        if len(heights) > 0:
            source_summary["img_h_mean"] = float(np.mean(heights))

        summary[source] = source_summary

    return summary


def save_quarantine_files(project_root, rf_images, kg_images, rf_boxes):
    interim_root = project_root / "data" / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)

    empty_labels = []
    for row in rf_images:
        if row["label_bytes"] is None or row["label_bytes"] == 0:
            empty_labels.append(row)

    corrupted_images = []
    for row in kg_images:
        if row["ok"] == 0:
            corrupted_images.append(row)

    near_full_boxes = []
    for row in rf_boxes:
        if row["class_name"] == "menu-text" and row["bbox_area_frac"] is not None:
            if row["bbox_area_frac"] >= 0.95:
                near_full_boxes.append(row)

    save_csv(interim_root / "quarantine_empty_labels.csv", empty_labels)
    save_csv(interim_root / "quarantine_corrupted_images.csv", corrupted_images)
    save_csv(interim_root / "quarantine_near_full_boxes.csv", near_full_boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args()

    if args.project_root is None:
        project_root = get_project_root()
    else:
        project_root = Path(args.project_root).resolve()

    interim_root = project_root / "data" / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)

    rf_images, rf_boxes, class_names = scan_roboflow(project_root)
    kg_images = scan_kaggle(project_root)

    save_csv(interim_root / "roboflow_manifest.csv", rf_images)
    save_csv(interim_root / "roboflow_boxes.csv", rf_boxes)
    save_csv(interim_root / "kaggle_manifest.csv", kg_images)
    save_quarantine_files(project_root, rf_images, kg_images, rf_boxes)

    summary = make_summary(rf_images + kg_images, rf_boxes)
    summary["roboflow"]["class_names"] = class_names

    summary_path = interim_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved files:")
    print(interim_root / "roboflow_manifest.csv")
    print(interim_root / "roboflow_boxes.csv")
    print(interim_root / "kaggle_manifest.csv")
    print(interim_root / "quarantine_empty_labels.csv")
    print(interim_root / "quarantine_corrupted_images.csv")
    print(interim_root / "quarantine_near_full_boxes.csv")
    print(summary_path)


if __name__ == "__main__":
    main()

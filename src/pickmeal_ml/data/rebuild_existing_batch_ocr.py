from pathlib import Path
import argparse

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def load_easyocr():
    try:
        import easyocr
        return easyocr
    except Exception:
        return None


def run_ocr_on_image(reader, image_path):
    result = reader.readtext(str(image_path), detail=1, paragraph=False)

    rows = []
    for line_index, item in enumerate(result, start=1):
        bbox, text, confidence = item

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        rows.append(
            {
                "line_order": line_index,
                "line_text": str(text).strip(),
                "ocr_confidence": float(confidence),
                "bbox_x1": float(min(xs)),
                "bbox_y1": float(min(ys)),
                "bbox_x2": float(max(xs)),
                "bbox_y2": float(max(ys)),
            }
        )

    return rows


def find_batch_dirs(annotation_root):
    return sorted([path for path in annotation_root.glob("batch_*") if path.is_dir()])


def detect_languages(lang_arg):
    if lang_arg is None or lang_arg.strip() == "":
        return ["en"]

    langs = [x.strip() for x in lang_arg.split(",") if x.strip()]
    return langs if langs else ["en"]


def rebuild_one_batch(project_root, batch_dir, reader):
    batch_name = batch_dir.name
    pages_path = batch_dir / f"{batch_name}_pages.csv"
    images_dir = batch_dir / "images"
    ocr_path = batch_dir / f"{batch_name}_ocr.csv"

    if not pages_path.exists():
        print(f"Skip {batch_name}: pages file not found")
        return

    if not images_dir.exists():
        print(f"Skip {batch_name}: images folder not found")
        return

    pages_df = pd.read_csv(pages_path)
    ocr_rows = []

    for _, row in pages_df.iterrows():
        menu_id = str(row.get("menu_id", "")).strip()
        page_id = str(row.get("page_id", "p1")).strip()
        image_file = str(row.get("image_file", "")).strip()

        if image_file == "":
            print(f"Skip row in {batch_name}: empty image_file")
            continue

        image_path = images_dir / image_file
        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        lines = run_ocr_on_image(reader, image_path)

        for line in lines:
            line_id = f"{menu_id}_l{line['line_order']:03d}"

            ocr_rows.append(
                {
                    "menu_id": menu_id,
                    "page_id": page_id,
                    "line_id": line_id,
                    "line_order": line["line_order"],
                    "line_text": line["line_text"],
                    "ocr_confidence": line["ocr_confidence"],
                    "bbox_x1": line["bbox_x1"],
                    "bbox_y1": line["bbox_y1"],
                    "bbox_x2": line["bbox_x2"],
                    "bbox_y2": line["bbox_y2"],
                    "image_file": image_file,
                    "image_path": str(image_path.relative_to(project_root)),
                }
            )

    ocr_df = pd.DataFrame(ocr_rows)
    ocr_df.to_csv(ocr_path, index=False)

    print(f"Saved OCR: {ocr_path}")
    print(f"Rows: {len(ocr_df)}")
    print()


def main(batch_name=None, langs=None):
    project_root = resolve_project_root()
    annotation_root = project_root / "data" / "processed" / "annotation_batches"

    easyocr_module = load_easyocr()
    if easyocr_module is None:
        raise ImportError(
            "easyocr is not installed. Install it with: pip install easyocr"
        )

    reader = easyocr_module.Reader(langs, gpu=False)

    batch_dirs = find_batch_dirs(annotation_root)
    if len(batch_dirs) == 0:
        raise FileNotFoundError(f"No batch_* folders found in {annotation_root}")

    for batch_dir in batch_dirs:
        if batch_name is not None and batch_dir.name != batch_name:
            continue

        rebuild_one_batch(project_root, batch_dir, reader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_name", type=str, default=None)
    parser.add_argument("--langs", type=str, default="en")
    args = parser.parse_args()

    main(batch_name=args.batch_name, langs=detect_languages(args.langs))

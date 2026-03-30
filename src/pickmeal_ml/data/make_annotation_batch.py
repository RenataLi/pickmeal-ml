from pathlib import Path
import argparse
import shutil

import pandas as pd


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


def make_menu_id(batch_name, index_value):
    return f"{batch_name}_{index_value:03d}"


def run_ocr_on_image(reader, image_path):
    result = reader.readtext(str(image_path), detail=1, paragraph=False)

    rows = []
    for line_idx, item in enumerate(result, start=1):
        bbox, text, conf = item

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]

        rows.append(
            {
                "line_order": line_idx,
                "line_text": str(text).strip(),
                "ocr_confidence": float(conf),
                "bbox_x1": float(min(xs)),
                "bbox_y1": float(min(ys)),
                "bbox_x2": float(max(xs)),
                "bbox_y2": float(max(ys)),
            }
        )

    return rows


def main(batch_name, n_pages, start_index, use_ocr):
    project_root = resolve_project_root()

    manifest_path = project_root / "data" / "interim" / "kaggle_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"File not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df = df[(df["source"] == "kaggle") & (df["ok"] == 1)].copy()
    df = df.sort_values("image_path").reset_index(drop=True)

    batch_df = df.iloc[start_index : start_index + n_pages].copy()
    if len(batch_df) == 0:
        raise ValueError("No images selected for the batch.")

    batch_root = project_root / "data" / "processed" / "annotation_batches" / batch_name
    image_out_dir = batch_root / "images"
    batch_root.mkdir(parents=True, exist_ok=True)
    image_out_dir.mkdir(parents=True, exist_ok=True)

    page_rows = []
    ocr_rows = []

    easyocr_module = None
    reader = None

    if use_ocr:
        easyocr_module = load_easyocr()
        if easyocr_module is None:
            print("easyocr is not installed. OCR file will be created as an empty template.")
            use_ocr = False
        else:
            reader = easyocr_module.Reader(["en"], gpu=False)

    for idx, (_, row) in enumerate(batch_df.iterrows(), start=1):
        menu_id = make_menu_id(batch_name, idx)
        page_id = "p1"

        src_image = project_root / row["image_path"]
        dst_image = image_out_dir / src_image.name

        shutil.copy2(src_image, dst_image)

        page_rows.append(
            {
                "menu_id": menu_id,
                "page_id": page_id,
                "image_file": dst_image.name,
                "image_path": str(dst_image.relative_to(project_root)),
                "source_image_path": row["image_path"],
                "img_w": row["img_w"],
                "img_h": row["img_h"],
            }
        )

        if use_ocr:
            lines = run_ocr_on_image(reader, dst_image)

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
                        "image_file": dst_image.name,
                        "image_path": str(dst_image.relative_to(project_root)),
                    }
                )

    pages_path = batch_root / f"{batch_name}_pages.csv"
    ocr_path = batch_root / f"{batch_name}_ocr.csv"

    pd.DataFrame(page_rows).to_csv(pages_path, index=False)

    if len(ocr_rows) == 0:
        pd.DataFrame(
            columns=[
                "menu_id",
                "page_id",
                "line_id",
                "line_order",
                "line_text",
                "ocr_confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "image_file",
                "image_path",
            ]
        ).to_csv(ocr_path, index=False)
    else:
        pd.DataFrame(ocr_rows).to_csv(ocr_path, index=False)

    print("Saved:")
    print(pages_path)
    print(ocr_path)
    print(image_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_name", type=str, default="batch_001")
    parser.add_argument("--n_pages", type=int, default=15)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--use_ocr", action="store_true")
    args = parser.parse_args()

    main(
        batch_name=args.batch_name,
        n_pages=args.n_pages,
        start_index=args.start_index,
        use_ocr=args.use_ocr,
    )
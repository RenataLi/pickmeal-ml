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


def load_manifest(project_root, source_name):
    manifest_path = project_root / "data" / "interim" / "kaggle_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"File not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df = df[df["source"] == source_name].copy()
    df = df[df["ok"] == 1].copy()
    df = df[df["is_tiny"] == 0].copy()
    df = df.dropna(subset=["image_path", "img_w", "img_h", "aspect_ratio", "img_area"]).copy()
    return df.reset_index(drop=True)


def load_used_source_images(project_root):
    annotation_root = project_root / "data" / "processed" / "annotation_batches"
    used_paths = set()

    if not annotation_root.exists():
        return used_paths

    for path in annotation_root.glob("batch_*/*_pages.csv"):
        df = pd.read_csv(path)
        if "source_image_path" not in df.columns:
            continue
        values = df["source_image_path"].dropna().astype(str).str.strip()
        used_paths.update(x for x in values if x != "")

    return used_paths


def add_buckets(df):
    out = df.copy()

    out["aspect_bucket"] = pd.cut(
        out["aspect_ratio"],
        bins=[0, 0.6, 0.75, 0.95, 1.2, 10],
        labels=["very_tall", "tall", "balanced", "wide", "very_wide"],
        include_lowest=True,
    ).astype(str)

    quantiles = out["img_area"].quantile([0.25, 0.5, 0.75]).tolist()
    q1, q2, q3 = quantiles
    bins = [-1, q1, q2, q3, float("inf")]
    labels = ["small", "medium", "large", "xlarge"]
    out["area_bucket"] = pd.cut(
        out["img_area"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    out["layout_bucket"] = out["aspect_bucket"] + "__" + out["area_bucket"]
    out["selection_score"] = (
        out["img_area"].rank(pct=True).fillna(0.0) * 0.55
        + out["image_bytes"].rank(pct=True).fillna(0.0) * 0.25
        + out["aspect_ratio"].rank(pct=True).fillna(0.0) * 0.20
    )
    return out


def choose_diverse_candidates(df, n_pages, random_state):
    if len(df) == 0:
        raise ValueError("No candidate pages available after filtering.")

    ranked = (
        df.sample(frac=1.0, random_state=random_state)
        .sort_values(
            ["layout_bucket", "selection_score", "img_area", "image_bytes", "image_path"],
            ascending=[True, False, False, False, True],
        )
        .reset_index(drop=True)
    )

    groups = {}
    for layout_bucket, group_df in ranked.groupby("layout_bucket", sort=True):
        groups[layout_bucket] = group_df.reset_index(drop=True)

    selected_rows = []
    selected_paths = set()
    selected_hashes = set()
    pointers = {key: 0 for key in groups}

    group_order = sorted(
        groups,
        key=lambda key: (
            len(groups[key]),
            -float(groups[key]["selection_score"].max()),
            key,
        ),
    )

    while len(selected_rows) < n_pages:
        added_in_round = False

        for key in group_order:
            group_df = groups[key]
            pointer = pointers[key]

            while pointer < len(group_df):
                row = group_df.iloc[pointer]
                pointer += 1

                image_path = str(row["image_path"])
                dhash = str(row.get("dhash", ""))

                if image_path in selected_paths:
                    continue

                if dhash != "" and dhash in selected_hashes:
                    continue

                selected_rows.append(row.to_dict())
                selected_paths.add(image_path)
                if dhash != "":
                    selected_hashes.add(dhash)
                pointers[key] = pointer
                added_in_round = True
                break

            pointers[key] = pointer

            if len(selected_rows) >= n_pages:
                break

        if added_in_round:
            continue

        remaining = ranked[~ranked["image_path"].astype(str).isin(selected_paths)].copy()
        if len(remaining) == 0:
            break

        remaining = remaining.sort_values(
            ["selection_score", "img_area", "image_bytes", "image_path"],
            ascending=[False, False, False, True],
        )

        for _, row in remaining.iterrows():
            image_path = str(row["image_path"])
            dhash = str(row.get("dhash", ""))
            if image_path in selected_paths:
                continue
            if dhash != "" and dhash in selected_hashes:
                continue

            selected_rows.append(row.to_dict())
            selected_paths.add(image_path)
            if dhash != "":
                selected_hashes.add(dhash)
            break
        else:
            break

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    if len(selected) == 0:
        raise ValueError("No pages were selected for the batch.")

    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["selection_reason"] = (
        "diverse layout bucket="
        + selected["layout_bucket"].astype(str)
        + "; aspect="
        + selected["aspect_bucket"].astype(str)
        + "; area="
        + selected["area_bucket"].astype(str)
    )
    return selected


def build_batch_outputs(project_root, batch_name, selected_df, use_ocr, ocr_languages):
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
            reader = easyocr_module.Reader(ocr_languages, gpu=False)

    for idx, (_, row) in enumerate(selected_df.iterrows(), start=1):
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
                "aspect_ratio": row["aspect_ratio"],
                "img_area": row["img_area"],
                "image_bytes": row["image_bytes"],
                "aspect_bucket": row["aspect_bucket"],
                "area_bucket": row["area_bucket"],
                "layout_bucket": row["layout_bucket"],
                "selection_rank": row["selection_rank"],
                "selection_reason": row["selection_reason"],
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
    candidate_path = batch_root / f"{batch_name}_gpt_candidates.csv"
    summary_path = batch_root / f"{batch_name}_selection_summary.csv"

    pages_df = pd.DataFrame(page_rows)
    pages_df.to_csv(pages_path, index=False)
    pages_df.to_csv(candidate_path, index=False)

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

    summary_df = (
        pages_df.groupby(["aspect_bucket", "area_bucket", "layout_bucket"], dropna=False)
        .size()
        .rename("n_pages")
        .reset_index()
        .sort_values(["n_pages", "layout_bucket"], ascending=[False, True])
    )
    summary_df.to_csv(summary_path, index=False)

    print("Saved:")
    print(pages_path)
    print(candidate_path)
    print(ocr_path)
    print(summary_path)
    print(image_out_dir)


def main(batch_name, n_pages, use_ocr, source_name, ocr_languages):
    project_root = resolve_project_root()
    manifest_df = load_manifest(project_root, source_name=source_name)
    used_paths = load_used_source_images(project_root)

    candidate_df = manifest_df[~manifest_df["image_path"].isin(used_paths)].copy()
    candidate_df = add_buckets(candidate_df)

    selected_df = choose_diverse_candidates(
        df=candidate_df,
        n_pages=n_pages,
        random_state=42,
    )

    build_batch_outputs(
        project_root=project_root,
        batch_name=batch_name,
        selected_df=selected_df,
        use_ocr=use_ocr,
        ocr_languages=ocr_languages,
    )

    print()
    print("Candidate pool after filtering:", len(candidate_df))
    print("Pages selected:", len(selected_df))
    print()
    print(selected_df[["selection_rank", "image_path", "layout_bucket", "selection_reason"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_name", type=str, default="batch_004")
    parser.add_argument("--n_pages", type=int, default=24)
    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument("--source_name", type=str, default="kaggle")
    parser.add_argument("--ocr_languages", nargs="+", default=["en"])
    args = parser.parse_args()

    main(
        batch_name=args.batch_name,
        n_pages=args.n_pages,
        use_ocr=args.use_ocr,
        source_name=args.source_name,
        ocr_languages=args.ocr_languages,
    )

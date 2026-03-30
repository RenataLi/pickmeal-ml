from pathlib import Path
import argparse
import shutil

import pandas as pd
import yaml


def resolve_project_root():
    root = Path.cwd().resolve()

    if (root / "data").exists():
        return root

    if (root.parent / "data").exists():
        return root.parent

    raise FileNotFoundError("Project root with data/ was not found.")


def read_class_names(data_yaml_path):
    text = data_yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]

    return list(names)


def keep_image(row, drop_empty_labels=False):
    if int(row["ok"]) != 1:
        return False

    if int(row["is_tiny"]) == 1:
        return False

    if pd.isna(row["label_path"]):
        return False

    if drop_empty_labels and float(row["label_bytes"]) == 0:
        return False

    return True


def main(drop_empty_labels=False):
    project_root = resolve_project_root()

    manifest_path = project_root / "data" / "interim" / "roboflow_manifest.csv"
    src_root = project_root / "data" / "raw" / "roboflow_menu_text_box_v3" / "Menu Text Box"

    if drop_empty_labels:
        out_root = project_root / "data" / "processed" / "roboflow_clean_yolo_drop_empty"
    else:
        out_root = project_root / "data" / "processed" / "roboflow_clean_yolo_keep_empty"

    df = pd.read_csv(manifest_path)

    if out_root.exists():
        shutil.rmtree(out_root)

    names = read_class_names(src_root / "data.yaml")

    total_kept = 0

    for split in ["train", "valid", "test"]:
        split_df = df[df["split"] == split].copy()
        split_df = split_df[split_df.apply(keep_image, axis=1, drop_empty_labels=drop_empty_labels)]

        kept_in_split = 0

        for _, row in split_df.iterrows():
            src_img = project_root / row["image_path"]
            src_lbl = project_root / row["label_path"]

            dst_img = out_root / split / "images" / src_img.name
            dst_lbl = out_root / split / "labels" / f"{src_img.stem}.txt"

            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)

            kept_in_split += 1

        total_kept += kept_in_split
        print(f"{split}: kept {kept_in_split} images")

    data_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }

    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print()
    print("Saved clean dataset to:", out_root)
    print("Saved yaml to:", yaml_path)
    print("Total kept images:", total_kept)
    print("drop_empty_labels:", drop_empty_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_empty_labels", action="store_true")
    args = parser.parse_args()

    main(drop_empty_labels=args.drop_empty_labels)
#!/usr/bin/env python3
"""
Process the labelled library folder into the shared dataset.

Walks raw/labelled/10X Libraries/<SpeciesDir>/ and extracts all chips from
every .tif file inside each species directory. The subdirectory name IS the
class label (e.g., the Closterium/ folder contains 7 sub-species TIFs,
all labelled "Closterium").

Behaviour:
  - New species → creates dataset/images/<species>/ automatically.
  - Existing species → continues chip numbering from the current highest ID
    so existing chips are never overwritten.
  - If any new chip is larger than the existing native_chip_size, all existing
    chips are re-padded to the new max_dim and dataset_info.json is updated.
  - New manifest rows are appended to dataset/manifest.csv.

Usage:
    python process_labelled.py
    python process_labelled.py --labelled_dir "raw/labelled/10X Libraries"
    python process_labelled.py --dataset_dir dataset --black_threshold 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Re-use extraction helpers from tif_montage_to_dataset
from tif_montage_to_dataset import extract_images, pad_to_size

VALID_EXTS = {".tif", ".tiff"}
MANIFEST_FIELDNAMES = ["chip_id", "image_path", "label", "source_collage_path",
                        "x", "y", "w", "h"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process labelled library TIFFs into the shared dataset."
    )
    parser.add_argument(
        "--labelled_dir",
        type=Path,
        default=Path("raw/labelled/10X Libraries"),
        help="Directory whose sub-folders are species labels. "
             "Default: raw/labelled/10X Libraries",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory (must contain manifest.csv). Default: dataset",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=10,
        help="Pixels with max channel value <= this are treated as black. Default: 10",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def next_chip_index(label_dir: Path, label: str) -> int:
    """Return the next available chip index for a species folder."""
    if not label_dir.exists():
        return 1
    existing = [
        p.stem for p in label_dir.iterdir()
        if p.suffix == ".png" and p.stem.startswith(label + "_")
    ]
    if not existing:
        return 1
    indices = []
    for stem in existing:
        try:
            indices.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 1


# ---------------------------------------------------------------------------
# Re-pad existing chips to a new (larger) max_dim
# ---------------------------------------------------------------------------

def repatch_existing_chips(
    existing_rows: list[dict],
    dataset_dir: Path,
    old_dim: int,
    new_dim: int,
) -> None:
    """Re-save all existing chips with the larger square padding."""
    print(f"\nExisting chips are {old_dim}px — re-padding to {new_dim}px ...")
    for row in existing_rows:
        img_path = dataset_dir / row["image_path"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            arr = np.array(im.convert("RGB"))
        arr = pad_to_size(arr, new_dim, new_dim)
        Image.fromarray(arr).save(img_path)
    print(f"  Re-padded {len(existing_rows)} existing chips.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.labelled_dir.exists():
        raise FileNotFoundError(f"Labelled directory not found: {args.labelled_dir}")

    # ------------------------------------------------------------------
    # Load existing dataset state
    # ------------------------------------------------------------------
    info_path = args.dataset_dir / "dataset_info.json"
    existing_native_dim = 0
    if info_path.exists():
        with info_path.open(encoding="utf-8") as f:
            info = json.load(f)
        existing_native_dim = info.get("native_chip_size", 0)
        print(f"Existing dataset native chip size: {existing_native_dim}px")
    else:
        print("No dataset_info.json found — treating as fresh dataset.")

    manifest_path = args.dataset_dir / "manifest.csv"
    existing_rows = read_manifest(manifest_path)
    print(f"Existing manifest: {len(existing_rows)} rows")

    # ------------------------------------------------------------------
    # Discover all (label, tif_path) pairs
    # ------------------------------------------------------------------
    species_dirs = sorted(
        d for d in args.labelled_dir.iterdir() if d.is_dir()
    )
    if not species_dirs:
        raise FileNotFoundError(f"No species subdirectories found in {args.labelled_dir}")

    # Build list: (label, tif_path)
    tif_entries: list[tuple[str, Path]] = []
    for species_dir in species_dirs:
        label = species_dir.name
        tifs = sorted(p for p in species_dir.iterdir() if p.suffix.lower() in VALID_EXTS)
        for tif_path in tifs:
            tif_entries.append((label, tif_path))

    print(f"\nFound {len(tif_entries)} TIF files across {len(species_dirs)} species")

    # ------------------------------------------------------------------
    # First pass: extract all chips, find new global max_dim
    # ------------------------------------------------------------------
    # Structure: { label: [(tif_path, chips), ...] }
    by_label: dict[str, list[tuple[Path, list]]] = {}
    new_max_h = 0
    new_max_w = 0

    for label, tif_path in tif_entries:
        print(f"  Extracting: {label} / {tif_path.name}")
        with Image.open(tif_path) as im:
            arr = np.array(im.convert("RGB"))
        chips = extract_images(arr, args.black_threshold)
        print(f"    → {len(chips)} chips")

        for img_arr, *_ in chips:
            new_max_h = max(new_max_h, img_arr.shape[0])
            new_max_w = max(new_max_w, img_arr.shape[1])

        by_label.setdefault(label, []).append((tif_path, chips))

    new_max_dim = max(new_max_h, new_max_w)
    global_max_dim = max(existing_native_dim, new_max_dim)
    print(f"\nNew data max chip size: {new_max_dim}px")
    print(f"Global max dim (all data): {global_max_dim}px")

    # ------------------------------------------------------------------
    # Re-pad existing chips if new data is larger
    # ------------------------------------------------------------------
    if global_max_dim > existing_native_dim and existing_rows:
        repatch_existing_chips(existing_rows, args.dataset_dir, existing_native_dim, global_max_dim)

    # ------------------------------------------------------------------
    # Second pass: save new chips with incremental IDs, collect manifest rows
    # ------------------------------------------------------------------
    images_root = args.dataset_dir / "images"
    new_manifest_rows: list[dict] = []
    total_saved = 0

    for label, tif_list in sorted(by_label.items()):
        label_dir = images_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        start_idx = next_chip_index(label_dir, label)
        chip_counter = start_idx

        for tif_path, chips in tif_list:
            for img_arr, x, y, w, h in chips:
                img_arr = pad_to_size(img_arr, global_max_dim, global_max_dim)
                chip_id = f"{label}_{chip_counter}"
                rel_path = f"images/{label}/{chip_id}.png"
                out_path = args.dataset_dir / rel_path
                Image.fromarray(img_arr).save(out_path)
                new_manifest_rows.append({
                    "chip_id":             chip_id,
                    "image_path":          rel_path,
                    "label":               label,
                    "source_collage_path": str(tif_path),
                    "x": x, "y": y, "w": w, "h": h,
                })
                chip_counter += 1

        added = chip_counter - start_idx
        total_saved += added
        status = "updated" if start_idx > 1 else "new"
        print(f"  {label:<25s} {added:>4} chips added  (idx {start_idx}–{chip_counter-1})  [{status}]")

    # ------------------------------------------------------------------
    # Write updated manifest
    # ------------------------------------------------------------------
    all_rows = existing_rows + new_manifest_rows
    write_manifest(all_rows, manifest_path)
    print(f"\nManifest updated: {len(existing_rows)} existing + {len(new_manifest_rows)} new = {len(all_rows)} total rows")

    # ------------------------------------------------------------------
    # Update dataset_info.json
    # ------------------------------------------------------------------
    if info_path.exists():
        with info_path.open(encoding="utf-8") as f:
            info = json.load(f)

        # Merge class lists
        existing_classes = set(info.get("classes", []))
        all_classes = sorted(existing_classes | set(by_label.keys()))
        class_to_idx = {c: i for i, c in enumerate(all_classes)}

        info["native_chip_size"] = global_max_dim
        info["input_size"] = max(global_max_dim, info.get("input_size", 0))
        info["num_classes"] = len(all_classes)
        info["classes"] = all_classes
        info["class_to_idx"] = class_to_idx
        # Clear rare_classes and split_counts — build_dataset.py will recompute
        info["rare_classes"] = []
        info["split_counts"] = {}
        info["note"] = (
            "dataset_info.json updated by process_labelled.py. "
            "Re-run build_dataset.py to refresh splits and rare_classes."
        )

        with info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"\ndataset_info.json updated → {len(all_classes)} classes, {global_max_dim}px chips")

    print(f"\nDone. {total_saved} new chips saved to '{images_root}/'")
    print("Run build_dataset.py to regenerate train/val/test splits.")


if __name__ == "__main__":
    main()

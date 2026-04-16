#!/usr/bin/env python3
"""
Convert montage-style .tif files (one per species class) into individual images.

Each .tif contains images of one algae species arranged in rows on a black background.
Row boundaries are detected via fully-black horizontal strips; column boundaries are
detected via fully-black vertical strips within each row band.

Class name is derived from the filename: everything before the first underscore.

All output images are padded with black pixels to the global max dimension so every
image in the dataset is square and uniform in size (max_dim x max_dim).

Outputs:
    dataset/images/<label>/<label>_<number>.png
    dataset/manifest.csv  (chip_id, image_path, label, source_collage_path, x, y, w, h)

Usage:
    python tif_montage_to_dataset.py
    python tif_montage_to_dataset.py --input_dir test_data --output_dir dataset
    python tif_montage_to_dataset.py --black_threshold 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split montage-style TIFFs into individual images."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing one .tif/.tiff per class. Default: test_data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset"),
        help="Root output directory. Images go to <output_dir>/images/<label>/. Default: dataset",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=10,
        help="Pixels with max channel value <= this are treated as black. Default: 10",
    )
    return parser.parse_args()


def find_bands(values: np.ndarray, black_threshold: int) -> list[tuple[int, int]]:
    """
    Given a 1-D array of max pixel values per row or column, return a list of
    (start, end) index pairs (inclusive) for contiguous non-black bands.
    """
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i, v in enumerate(values):
        if v > black_threshold and not in_band:
            in_band = True
            start = i
        elif v <= black_threshold and in_band:
            in_band = False
            bands.append((start, i - 1))
    if in_band:
        bands.append((start, len(values) - 1))
    return bands


def trim_black(crop: np.ndarray, black_threshold: int) -> np.ndarray:
    """Remove black rows and columns from all four edges of a crop."""
    row_max = crop.max(axis=(1, 2))
    col_max = crop.max(axis=(0, 2))

    non_black_rows = np.where(row_max > black_threshold)[0]
    non_black_cols = np.where(col_max > black_threshold)[0]

    if non_black_rows.size == 0 or non_black_cols.size == 0:
        return crop  # entirely black — return as-is

    r0, r1 = int(non_black_rows[0]), int(non_black_rows[-1])
    c0, c1 = int(non_black_cols[0]), int(non_black_cols[-1])
    return crop[r0 : r1 + 1, c0 : c1 + 1]


def pad_to_size(crop: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Pad the crop with black pixels on the right and bottom to exactly
    target_h x target_w. The crop must not exceed the target dimensions.
    """
    h, w = crop.shape[:2]
    channels = crop.shape[2] if crop.ndim == 3 else 1

    if h == target_h and w == target_w:
        return crop

    canvas = np.zeros((target_h, target_w, channels), dtype=crop.dtype)
    canvas[:h, :w] = crop
    return canvas


def extract_images(
    arr: np.ndarray, black_threshold: int
) -> list[tuple[np.ndarray, int, int, int, int]]:
    """
    Extract all individual images from a montage array.

    Strategy:
    1. Find row bands (separated by black rows).
    2. Within each row band, find column segments (separated by black columns).
    3. Trim residual black from each crop.

    Returns a list of (chip_array, x, y, w, h) where x/y/w/h are the
    chip's bounding box in the original montage coordinate space.
    """
    row_max = arr.max(axis=(1, 2))
    row_bands = find_bands(row_max, black_threshold)

    chips: list[tuple[np.ndarray, int, int, int, int]] = []
    for r_start, r_end in row_bands:
        row_band = arr[r_start : r_end + 1]

        col_max = row_band.max(axis=(0, 2))
        col_segs = find_bands(col_max, black_threshold)

        for c_start, c_end in col_segs:
            crop = row_band[:, c_start : c_end + 1]

            # Compute trim offsets so we can record the exact bbox in the
            # original montage coordinates.
            row_max_crop = crop.max(axis=(1, 2))
            col_max_crop = crop.max(axis=(0, 2))
            non_black_rows = np.where(row_max_crop > black_threshold)[0]
            non_black_cols = np.where(col_max_crop > black_threshold)[0]

            if non_black_rows.size == 0 or non_black_cols.size == 0:
                continue  # entirely black — skip

            r0, r1 = int(non_black_rows[0]), int(non_black_rows[-1])
            c0, c1 = int(non_black_cols[0]), int(non_black_cols[-1])
            trimmed = crop[r0 : r1 + 1, c0 : c1 + 1]

            if trimmed.size > 0:
                orig_x = c_start + c0
                orig_y = r_start + r0
                orig_w = c1 - c0 + 1
                orig_h = r1 - r0 + 1
                chips.append((trimmed, orig_x, orig_y, orig_w, orig_h))

    return chips


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    tif_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in VALID_EXTS
    )
    if not tif_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {args.input_dir}")

    # First pass: extract all chips and track global max dimension for square padding.
    # Each entry: (class_name, tif_path, chips) where chips = [(arr, x, y, w, h), ...]
    all_classes: list[tuple[str, Path, list[tuple[np.ndarray, int, int, int, int]]]] = []
    max_h = 0
    max_w = 0

    for tif_path in tif_paths:
        class_name = tif_path.stem.split("_")[0]
        print(f"Processing: {tif_path.name}  →  class '{class_name}'")

        with Image.open(tif_path) as im:
            arr = np.array(im.convert("RGB"))

        chips = extract_images(arr, args.black_threshold)
        print(f"  Found {len(chips)} chips")

        for img_arr, *_ in chips:
            max_h = max(max_h, img_arr.shape[0])
            max_w = max(max_w, img_arr.shape[1])

        all_classes.append((class_name, tif_path, chips))

    MAX_DIM_CAP = 600
    max_dim = min(max(max_h, max_w), MAX_DIM_CAP)
    print(f"\nPadding all images to {max_dim}x{max_dim} (square, capped at {MAX_DIM_CAP})")

    # Second pass: pad to max_dim x max_dim, save to dataset/images/<label>/, write manifest.
    images_root = args.output_dir / "images"
    manifest_path = args.output_dir / "manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    manifest_rows: list[dict] = []

    for class_name, tif_path, chips in all_classes:
        label_dir = images_root / class_name
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_arr, x, y, w, h) in enumerate(chips, start=1):
            # Scale down if the chip exceeds the cap, then pad to square.
            ch, cw = img_arr.shape[:2]
            if max(ch, cw) > max_dim:
                scale = max_dim / max(ch, cw)
                new_cw, new_ch = int(cw * scale), int(ch * scale)
                img_pil = Image.fromarray(img_arr).resize((new_cw, new_ch), Image.LANCZOS)
                img_arr = np.array(img_pil)
            img_arr = pad_to_size(img_arr, max_dim, max_dim)
            chip_id = f"{class_name}_{idx}"
            rel_path = f"images/{class_name}/{chip_id}.png"
            out_path = args.output_dir / rel_path
            Image.fromarray(img_arr).save(out_path)
            manifest_rows.append({
                "chip_id": chip_id,
                "image_path": rel_path,
                "label": class_name,
                "source_collage_path": str(tif_path),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            })

        total_saved += len(chips)

    fieldnames = ["chip_id", "image_path", "label", "source_collage_path", "x", "y", "w", "h"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Done. Saved {total_saved} images to '{images_root}/'")
    print(f"Manifest written to '{manifest_path}' ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert montage-style .tif files (one per species class) into individual images.

Each .tif contains images of one algae species arranged in rows on a black background.
Row boundaries are detected via fully-black horizontal strips; column boundaries are
detected via fully-black vertical strips within each row band.

Class name is derived from the filename: everything before the first underscore.

Output: output_data/<classname>_<number>.png

Usage:
    python tif_montage_to_dataset.py
    python tif_montage_to_dataset.py --input_dir test_data --output_dir output_data
    python tif_montage_to_dataset.py --black_threshold 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTS = {".tif", ".tiff"}

# Size to pad output images to. If an image is smaller than MAX_SIZE in either
# dimension, black pixels are added on the right and/or bottom. Images larger
# than MAX_SIZE are kept at their natural size. Set to None to skip padding.
MAX_SIZE: int | None = 128


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
        default=Path("output_data"),
        help="Directory where individual images will be written. Default: output_data",
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


def pad_to_max_size(crop: np.ndarray, max_size: int) -> np.ndarray:
    """
    Pad the crop with black pixels on the right and bottom so that its
    dimensions are at least max_size x max_size. If already larger, keep as-is.
    """
    h, w = crop.shape[:2]
    channels = crop.shape[2] if crop.ndim == 3 else 1
    target_h = max(h, max_size)
    target_w = max(w, max_size)

    if h == target_h and w == target_w:
        return crop

    canvas = np.zeros((target_h, target_w, channels), dtype=crop.dtype)
    canvas[:h, :w] = crop
    return canvas


def extract_images(
    arr: np.ndarray, black_threshold: int
) -> list[np.ndarray]:
    """
    Extract all individual images from a montage array.

    Strategy:
    1. Find row bands (separated by black rows).
    2. Within each row band, find column segments (separated by black columns).
    3. Trim residual black from each crop.
    """
    row_max = arr.max(axis=(1, 2))
    row_bands = find_bands(row_max, black_threshold)

    images: list[np.ndarray] = []
    for r_start, r_end in row_bands:
        row_band = arr[r_start : r_end + 1]

        col_max = row_band.max(axis=(0, 2))
        col_segs = find_bands(col_max, black_threshold)

        for c_start, c_end in col_segs:
            crop = row_band[:, c_start : c_end + 1]
            crop = trim_black(crop, black_threshold)
            if crop.size > 0:
                images.append(crop)

    return images


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    tif_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in VALID_EXTS
    )
    if not tif_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for tif_path in tif_paths:
        class_name = tif_path.stem.split("_")[0]
        print(f"Processing: {tif_path.name}  →  class '{class_name}'")

        with Image.open(tif_path) as im:
            arr = np.array(im.convert("RGB"))

        images = extract_images(arr, args.black_threshold)
        print(f"  Found {len(images)} images")

        for idx, img_arr in enumerate(images, start=1):
            if MAX_SIZE is not None:
                img_arr = pad_to_max_size(img_arr, MAX_SIZE)

            out_path = args.output_dir / f"{class_name}_{idx}.png"
            Image.fromarray(img_arr).save(out_path)

        total_saved += len(images)

    print(f"\nDone. Saved {total_saved} images to '{args.output_dir}/'")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
tif_montage_to_dataset.py — Unified chip extractor (replaces process_labelled.py).

Processes ALL labelled TIF sources in one pass and produces a clean dataset
ready for augment_dataset.py → build_dataset.py → train.py.

Sources processed (any that exist are included, missing ones are skipped):

  1. test_data/
       Flat directory of montage TIFs. Label derived from filename:
       "Anabaena_lib_images_000001.tif" → "Anabaena"

  2. raw/labelled/  (three sub-layouts handled automatically)
       a. Species-directory layout  (e.g. 10X Libraries/)
            raw/labelled/.../Anabaena/any.tif  →  label = "Anabaena"
       b. Flat-filename layout  (e.g. FlowCam Cyano Example/Libraries/)
            Libraries/Anabaena_lib_000001.tif  →  label = "Anabaena"
       c. Example-prefix layout  (e.g. Libraries - Freshwater Organisms/)
            Example_Anabaena-coiled_10X_TR_lib.tif  →  label = "Anabaena"

  Note: raw/labelled/Images of Fresh Water Algae/ contains whole-field JPGs
  (not montage TIFs) and is skipped — those images need separate handling.

Each organism chip is saved as a tight crop at its natural trimmed size.
No black padding is added. Very large chips (longest side > MAX_CHIP_DIM)
are scaled down proportionally. train.py's Resize handles variable sizes.

This is a FRESH BUILD: dataset/images/ and manifest.csv are rebuilt from
scratch every run. Augmented files (aug_*.png) are preserved.

Outputs:
    dataset/images/<label>/<label>_<N>.png
    dataset/manifest.csv

Usage:
    python tif_montage_to_dataset.py
    python tif_montage_to_dataset.py --test_data_dir test_data --labelled_dir raw/labelled
    python tif_montage_to_dataset.py --output_dir dataset --black_threshold 10
    python tif_montage_to_dataset.py --no_labelled    # test_data only
    python tif_montage_to_dataset.py --no_test_data   # raw/labelled only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


VALID_TIF_EXTS = {".tif", ".tiff"}
MAX_CHIP_DIM    = 600  # scale down if longest chip side exceeds this; no padding

# Directory names that are organiser containers, NOT species labels.
# A TIF whose immediate parent matches one of these uses filename-based
# label extraction instead of using the parent name as the label.
NON_SPECIES_DIRS: set[str] = {
    "Libraries",
    "Auto-Image Libraries",
    "Trigger Mode Libraries",
    "Filters",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified chip extractor: processes test_data/ and raw/labelled/ "
            "TIF sources into dataset/images/ with tight crops."
        ),
    )
    parser.add_argument(
        "--test_data_dir", type=Path, default=Path("test_data"),
        help="Flat directory of montage TIFs (label from filename). Default: test_data",
    )
    parser.add_argument(
        "--labelled_dir", type=Path, default=Path("raw/labelled"),
        help="Root of multi-layout labelled TIF folders. Default: raw/labelled",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("dataset"),
        help="Root output directory. Default: dataset",
    )
    parser.add_argument(
        "--black_threshold", type=int, default=10,
        help="Max channel value considered black. Default: 10",
    )
    parser.add_argument(
        "--no_test_data", action="store_true",
        help="Skip test_data/ source.",
    )
    parser.add_argument(
        "--no_labelled", action="store_true",
        help="Skip raw/labelled/ source.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image extraction helpers
# ---------------------------------------------------------------------------

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


def extract_images(
    arr: np.ndarray, black_threshold: int
) -> list[tuple[np.ndarray, int, int, int, int]]:
    """
    Extract all individual organism chips from a montage array.

    1. Find row bands (separated by fully-black horizontal strips).
    2. Within each row band, find column segments (separated by fully-black
       vertical strips).
    3. Trim residual black from each crop.

    Returns list of (chip_array, x, y, w, h) where x/y/w/h are the chip's
    bounding box in the original montage coordinate space.
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

            row_max_crop = crop.max(axis=(1, 2))
            col_max_crop = crop.max(axis=(0, 2))
            non_black_rows = np.where(row_max_crop > black_threshold)[0]
            non_black_cols = np.where(col_max_crop > black_threshold)[0]

            if non_black_rows.size == 0 or non_black_cols.size == 0:
                continue

            r0, r1 = int(non_black_rows[0]), int(non_black_rows[-1])
            c0, c1 = int(non_black_cols[0]), int(non_black_cols[-1])
            trimmed = crop[r0 : r1 + 1, c0 : c1 + 1]

            if trimmed.size > 0:
                chips.append((trimmed, c_start + c0, r_start + r0, c1 - c0 + 1, r1 - r0 + 1))

    return chips


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

def label_from_filename(tif_path: Path) -> str:
    """
    Derive species label from the TIF filename stem.
    "Anabaena_lib_images_000001" → "Anabaena"
    "Cosmarium-ish_lib_images_000001" → "Cosmarium-ish"
    """
    return tif_path.stem.split("_")[0]


def label_from_example_prefix(stem: str) -> str:
    """
    Handle the Example_<Species>[-variant]_10X_... naming convention.
    "Example_Anabaena-coiled_10X_TR_lib_images_000001" → "Anabaena"
    "Example_Pediastrum-2_10X_TR_lib_images_000001"    → "Pediastrum"
    """
    without_prefix = stem[len("Example_"):]          # "Anabaena-coiled_10X_..."
    before_10x     = without_prefix.split("_10X")[0] # "Anabaena-coiled"
    return before_10x.split("-")[0]                  # "Anabaena"


def extract_label(tif_path: Path) -> str:
    """
    Determine the species label for a TIF under raw/labelled/.

    Rules (in priority order):
      1. If the immediate parent dir is NOT in NON_SPECIES_DIRS, use it as
         the label (species-directory layout: .../Anabaena/foo.tif).
      2. If the filename stem starts with "Example_", use example-prefix
         extraction.
      3. Otherwise use the first "_"-delimited token of the stem (flat-
         filename layout: Libraries/Anabaena_lib_000001.tif).
    """
    parent = tif_path.parent.name
    if parent not in NON_SPECIES_DIRS:
        return parent

    stem = tif_path.stem
    if stem.startswith("Example_"):
        return label_from_example_prefix(stem)

    return label_from_filename(tif_path)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def collect_test_data_tifs(test_data_dir: Path) -> list[tuple[str, Path]]:
    """
    Collect (label, tif_path) pairs from a flat test_data directory.
    Label is derived from the filename.
    """
    entries: list[tuple[str, Path]] = []
    for p in sorted(test_data_dir.iterdir()):
        if p.suffix.lower() in VALID_TIF_EXTS:
            entries.append((label_from_filename(p), p))
    return entries


def collect_labelled_tifs(labelled_dir: Path) -> list[tuple[str, Path]]:
    """
    Recursively collect (label, tif_path) pairs from all sub-layouts under
    raw/labelled/. JPG/PNG reference images are ignored.
    """
    entries: list[tuple[str, Path]] = []
    for p in sorted(labelled_dir.rglob("*")):
        if p.suffix.lower() in VALID_TIF_EXTS:
            entries.append((extract_label(p), p))
    return entries


# ---------------------------------------------------------------------------
# Chip saving
# ---------------------------------------------------------------------------

def save_chips(
    tif_entries: list[tuple[str, Path]],
    images_root: Path,
    dataset_dir: Path,
    black_threshold: int,
    chip_counters: dict[str, int],
) -> list[dict]:
    """
    Extract and save tight-crop chips from all (label, tif_path) entries.
    chip_counters tracks the next available index per species (modified in place).
    Returns a list of new manifest row dicts.
    """
    manifest_rows: list[dict] = []

    for label, tif_path in tif_entries:
        with Image.open(tif_path) as im:
            arr = np.array(im.convert("RGB"))

        chips = extract_images(arr, black_threshold)
        if not chips:
            print(f"  WARNING: no chips found in {tif_path.name} — skipping.")
            continue

        label_dir = images_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_arr, x, y, w, h in chips:
            ch, cw = img_arr.shape[:2]
            if max(ch, cw) > MAX_CHIP_DIM:
                scale = MAX_CHIP_DIM / max(ch, cw)
                new_cw = max(1, int(cw * scale))
                new_ch = max(1, int(ch * scale))
                img_arr = np.array(
                    Image.fromarray(img_arr).resize((new_cw, new_ch), Image.LANCZOS)
                )

            idx = chip_counters.get(label, 1)
            chip_id  = f"{label}_{idx}"
            rel_path = f"images/{label}/{chip_id}.png"
            Image.fromarray(img_arr).save(dataset_dir / rel_path)

            manifest_rows.append({
                "chip_id":             chip_id,
                "image_path":          rel_path,
                "label":               label,
                "source_collage_path": str(tif_path),
                "x": x, "y": y, "w": w, "h": h,
            })
            chip_counters[label] = idx + 1

    return manifest_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Discover all TIF sources                                             #
    # ------------------------------------------------------------------ #
    all_entries: list[tuple[str, Path]] = []

    if not args.no_test_data:
        if args.test_data_dir.exists():
            tds = collect_test_data_tifs(args.test_data_dir)
            print(f"test_data/        : {len(tds)} TIF(s)")
            all_entries.extend(tds)
        else:
            print(f"test_data/        : directory not found — skipped ({args.test_data_dir})")

    if not args.no_labelled:
        if args.labelled_dir.exists():
            labs = collect_labelled_tifs(args.labelled_dir)
            print(f"raw/labelled/     : {len(labs)} TIF(s)")
            all_entries.extend(labs)
        else:
            print(f"raw/labelled/     : directory not found — skipped ({args.labelled_dir})")

    if not all_entries:
        raise FileNotFoundError(
            "No TIF files found in any source directory. "
            "Check --test_data_dir and --labelled_dir."
        )

    from collections import Counter
    label_counts = Counter(label for label, _ in all_entries)
    print(f"\nTotal: {len(all_entries)} TIF file(s) across {len(label_counts)} species:")
    for lbl in sorted(label_counts):
        print(f"  {lbl:<25s} {label_counts[lbl]} TIF(s)")

    # ------------------------------------------------------------------ #
    # Prepare output directory — wipe original chips, keep aug_* files    #
    # ------------------------------------------------------------------ #
    images_root   = args.output_dir / "images"
    manifest_path = args.output_dir / "manifest.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    removed = 0
    for species_dir in images_root.iterdir():
        if not species_dir.is_dir():
            continue
        for png in species_dir.glob("*.png"):
            if not png.stem.startswith("aug_"):
                png.unlink()
                removed += 1
    if removed:
        print(f"\nCleared {removed} original chip(s) from previous run (aug_* preserved).")

    # ------------------------------------------------------------------ #
    # Extract and save chips                                               #
    # ------------------------------------------------------------------ #
    print(f"\nExtracting chips  (MAX_CHIP_DIM={MAX_CHIP_DIM}px, no padding) ...")
    chip_counters: dict[str, int] = {}
    manifest_rows = save_chips(
        tif_entries=all_entries,
        images_root=images_root,
        dataset_dir=args.output_dir,
        black_threshold=args.black_threshold,
        chip_counters=chip_counters,
    )

    # ------------------------------------------------------------------ #
    # Per-species summary                                                  #
    # ------------------------------------------------------------------ #
    per_species = Counter(r["label"] for r in manifest_rows)
    print(f"\nChips extracted   ({sum(per_species.values())} total):")
    for lbl in sorted(per_species):
        print(f"  {lbl:<25s} {per_species[lbl]}")

    # ------------------------------------------------------------------ #
    # Write manifest (preserve any surviving aug rows)                     #
    # ------------------------------------------------------------------ #
    aug_rows: list[dict] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("source_collage_path") == "augmented":
                    img_path = args.output_dir / row["image_path"]
                    if img_path.exists():
                        aug_rows.append(row)

    fieldnames = ["chip_id", "image_path", "label", "source_collage_path", "x", "y", "w", "h"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
        writer.writerows(aug_rows)

    total_rows = len(manifest_rows) + len(aug_rows)
    print(f"\nManifest written  : {manifest_path}")
    print(
        f"  {len(manifest_rows)} original chip rows  +  "
        f"{len(aug_rows)} preserved aug rows  =  {total_rows} total"
    )
    print(
        "\nNext steps:\n"
        "  python augment_dataset.py   # generate augmented chips for under-represented species\n"
        "  python build_dataset.py     # create train / val / test splits"
    )


if __name__ == "__main__":
    main()


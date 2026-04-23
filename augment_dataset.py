#!/usr/bin/env python3
"""
augment_dataset.py — Offline data augmentation for under-represented species.

For each species below the target image count, generates augmented copies using
the same transforms applied during training (minus ToTensor / Normalize, since
outputs are saved back to disk as PNG files). Species already at or above the
target are skipped entirely.

Augmentation multipliers are naturally curated by count: a species with 5
images gets far more passes than one with 180. The script cycles through
source images in sorted order so every original contributes equally.

A warning is printed when a species has so few source images that the
augmentation multiplier exceeds 10× — giving you visibility into the most
extreme cases.

Augmented files land in the same folder as the originals:
    dataset/images/<Species>/aug_0001_<original_stem>.png

All new augmented files are also registered in dataset/manifest.csv so that
build_dataset.py can pick them up when rebuilding the train/val/test splits.

The script is idempotent: re-running it skips image files that already exist
and manifest rows that are already registered.

After this script completes, re-run build_dataset.py to rebuild the
train / val / test splits with the augmented data included.

Usage:
    python augment_dataset.py
    python augment_dataset.py --target 200
    python augment_dataset.py --target 300 --dataset_dir dataset
    python augment_dataset.py --dry_run          # preview without writing
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation pipeline — mirrors train.py, omitting ToTensor and Normalize
# since we save back to disk as RGB PNG images rather than feeding a model.
# ---------------------------------------------------------------------------

def build_aug_transform() -> transforms.Compose:
    """
    Identical augmentation policy to train.py:
      - RandomHorizontalFlip   (organisms have no left/right preference)
      - RandomVerticalFlip     (organisms have no up/down preference)
      - RandomRotation(180)    (full 360° orientation is valid under microscope)
      - ColorJitter            (compensates for lighting / white-balance variation)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
    ])


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

MANIFEST_FIELDNAMES = ["chip_id", "image_path", "label", "source_collage_path", "x", "y", "w", "h"]


def read_manifest_ids(manifest_path: Path) -> set[str]:
    """Return the set of chip_ids already registered in manifest.csv."""
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return {row["chip_id"] for row in csv.DictReader(f)}


def append_manifest_rows(manifest_path: Path, new_rows: list[dict]) -> None:
    """Append rows to manifest.csv, creating the file with a header if needed."""
    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)


# ---------------------------------------------------------------------------
# Per-species augmentation
# ---------------------------------------------------------------------------

def augment_species(
    species_dir: Path,
    needed: int,
    transform: transforms.Compose,
    dry_run: bool,
    existing_ids: set[str],
) -> list[dict]:
    """
    Generate `needed` augmented images for one species directory.

    Source images are selected by cycling through the sorted list of originals
    so every original contributes equally regardless of how many are needed.
    Each call to transform() draws fresh random parameters, so even the same
    source image produces visually distinct outputs.

    Returns a list of manifest row dicts for images that were newly written
    (or that would be written in dry_run mode). Already-registered files and
    already-existing aug files are both skipped (idempotent).
    """
    sources = sorted(
        p for p in species_dir.glob("*.png")
        if not p.stem.startswith("aug_")
    )
    if not sources:
        print(f"  WARNING: no source PNG images found in {species_dir}, skipping.")
        return []

    multiplier = needed / len(sources)
    if multiplier > 10:
        print(
            f"  NOTE  {species_dir.name}: only {len(sources)} source image(s) — "
            f"augmentation multiplier = {multiplier:.1f}×. "
            f"Consider collecting more originals when possible."
        )

    species = species_dir.name
    new_rows: list[dict] = []

    for i in range(1, needed + 1):
        src_path = sources[(i - 1) % len(sources)]
        out_stem = f"aug_{i:04d}_{src_path.stem}"
        out_name = f"{out_stem}.png"
        out_path = species_dir / out_name
        chip_id  = out_stem

        # Skip if already registered in manifest (idempotent)
        if chip_id in existing_ids:
            continue

        if not dry_run:
            if not out_path.exists():
                img = Image.open(src_path).convert("RGB")
                aug_img = transform(img)
                aug_img.save(out_path)

        # Use forward slashes for image_path so it's consistent with the
        # existing manifest entries written by tif_montage_to_dataset.py
        rel_path = f"images/{species}/{out_name}"
        new_rows.append({
            "chip_id":             chip_id,
            "image_path":          rel_path,
            "label":               species,
            "source_collage_path": "augmented",
            "x": 0, "y": 0, "w": 0, "h": 0,
        })

    return new_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline augmentation: bring all species up to a minimum total image "
            "count. Re-run build_dataset.py afterwards to rebuild splits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("dataset"),
        help="Root dataset directory (must contain images/). Default: dataset",
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help=(
            "Minimum total images per species after augmentation. "
            "With the default 70%% train split this yields ~140 training images. "
            "Raise to 215 for ~150 training images. Default: 200"
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the augmentation plan without writing any files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    args = parser.parse_args()

    # Seed all relevant random sources for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    images_dir = args.dataset_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Make sure --dataset_dir points to the correct dataset root."
        )

    manifest_path = args.dataset_dir / "manifest.csv"

    transform = build_aug_transform()

    # Load existing manifest to know what's already registered.
    # current_count is derived from the manifest (what build_dataset.py sees),
    # NOT from the filesystem — so aug files on disk but not yet registered
    # are treated as missing and will be registered.
    existing_ids = read_manifest_ids(manifest_path)
    print(f"Manifest            : {len(existing_ids)} existing entries in {manifest_path}")

    # Per-species manifest counts (originals + already-registered aug rows)
    manifest_label_counts: dict[str, int] = {}
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row["label"]
                manifest_label_counts[lbl] = manifest_label_counts.get(lbl, 0) + 1

    # Collect species dirs; sort ascending by manifest count so the
    # most-needy species are processed (and printed) first.
    species_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
    counts: list[tuple[int, Path]] = [
        (manifest_label_counts.get(sd.name, 0), sd)
        for sd in species_dirs
    ]
    # Sort ascending so the most-needy species are processed first
    counts.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    # Report header                                                        #
    # ------------------------------------------------------------------ #
    col_w = 24
    header = f"{'Species':<{col_w}} {'Current':>8} {'To add':>8} {'New total':>10}"
    sep    = "-" * len(header)

    print(f"\nAugmentation target : {args.target} total images per species")
    print(f"  → ~{int(args.target * 0.70)} training images at 70% split")
    print(f"Dataset images dir  : {images_dir.resolve()}")
    if args.dry_run:
        print("Mode                : DRY RUN — no files will be written\n")
    else:
        print("Mode                : LIVE — augmented images will be saved\n")

    print(header)
    print(sep)

    total_generated = 0
    skipped_count   = 0
    all_new_rows: list[dict] = []

    for current_count, sd in counts:
        needed = max(0, args.target - current_count)

        if needed == 0:
            skipped_count += 1
            print(f"{sd.name:<{col_w}} {current_count:>8} {'—':>8} {'(at target)':>10}")
            continue

        new_total = current_count + needed
        print(f"{sd.name:<{col_w}} {current_count:>8} {needed:>8} {new_total:>10}")

        new_rows = augment_species(
            species_dir=sd,
            needed=needed,
            transform=transform,
            dry_run=args.dry_run,
            existing_ids=existing_ids,
        )
        all_new_rows.extend(new_rows)
        total_generated += len(new_rows)

    print(sep)
    print(f"\n{skipped_count} species already at or above target — skipped.")

    action = "Would register" if args.dry_run else "Registered"
    print(f"{action} {total_generated} augmented image(s) in manifest.")

    if not args.dry_run and all_new_rows:
        append_manifest_rows(manifest_path, all_new_rows)
        print(
            "\nNext step:\n"
            "  python build_dataset.py\n"
            "to rebuild the train / val / test splits with the augmented "
            "images included."
        )
    elif args.dry_run and total_generated > 0:
        print(
            "\nRe-run without --dry_run to apply the augmentation."
        )


if __name__ == "__main__":
    main()

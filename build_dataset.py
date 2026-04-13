#!/usr/bin/env python3
"""
Phase 2 — Dataset build: stratified train/val/test split.

Reads dataset/manifest.csv, performs a stratified split, and writes:
    dataset/splits/train.csv
    dataset/splits/val.csv
    dataset/splits/test.csv
    dataset/dataset_info.json   —  records input_size, classes, per-split counts

Input size rule:
    - Read the native chip size from the images (max_dim x max_dim from Phase 1).
    - If native size < 224, set input_size = 224 (models will resize up during training).
    - Otherwise use native size as-is (all 4 models handle any size via AdaptiveAvgPool).

Usage:
    python build_dataset.py
    python build_dataset.py --dataset_dir dataset --train 0.70 --val 0.15 --test 0.15
    python build_dataset.py --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


MIN_INPUT_SIZE = 224  # resize up to this if native chips are smaller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split of the chip dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory containing manifest.csv and images/. Default: dataset",
    )
    parser.add_argument(
        "--train", type=float, default=0.70,
        help="Fraction of data for training. Default: 0.70",
    )
    parser.add_argument(
        "--val", type=float, default=0.15,
        help="Fraction of data for validation. Default: 0.15",
    )
    parser.add_argument(
        "--test", type=float, default=0.15,
        help="Fraction of data for testing. Default: 0.15",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    return parser.parse_args()


def png_size(path: Path) -> tuple[int, int]:
    """Read (width, height) from a PNG header without decoding pixel data."""
    with path.open("rb") as f:
        f.read(8)   # PNG signature
        f.read(4)   # IHDR chunk length
        f.read(4)   # 'IHDR' tag
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    return w, h


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--train + --val + --test must sum to 1.0, got {total:.4f}")

    manifest_path = args.dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run tif_montage_to_dataset.py first to generate the dataset."
        )

    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError("manifest.csv is empty.")

    # ------------------------------------------------------------------ #
    # Determine input size from native chip dimensions                     #
    # ------------------------------------------------------------------ #
    first_img = args.dataset_dir / rows[0]["image_path"]
    if not first_img.exists():
        raise FileNotFoundError(f"Image referenced in manifest not found: {first_img}")

    native_w, native_h = png_size(first_img)
    native_size = max(native_w, native_h)   # chips are square; be safe
    input_size = max(native_size, MIN_INPUT_SIZE)

    if native_size < MIN_INPUT_SIZE:
        print(
            f"Native chip size {native_size}px is smaller than {MIN_INPUT_SIZE}px. "
            f"Training will resize images up to {input_size}px."
        )
    else:
        print(f"Native chip size {native_size}px — using {input_size}px as input_size.")

    # ------------------------------------------------------------------ #
    # Class inventory                                                      #
    # ------------------------------------------------------------------ #
    labels = [r["label"] for r in rows]
    label_counts = Counter(labels)
    classes = sorted(label_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"\nDataset: {len(rows)} chips  |  {len(classes)} classes")
    for cls in classes:
        print(f"  {cls}: {label_counts[cls]} chips")

    # ------------------------------------------------------------------ #
    # Rare-class handling                                                  #
    # A class needs at least 2 samples in the val+test pool (one each).  #
    # Minimum total = ceil(2 / val_test_frac).  Classes below that       #
    # threshold go entirely into train; a warning is printed.             #
    # ------------------------------------------------------------------ #
    val_test_frac = args.val + args.test
    min_for_split = math.ceil(2 / val_test_frac)  # e.g. 7 for 70/15/15

    rare_classes = {c for c, n in label_counts.items() if n < min_for_split}
    if rare_classes:
        print(
            f"\nWARNING: {sorted(rare_classes)} have fewer than {min_for_split} chips "
            f"and cannot be stratified across all three splits.\n"
            f"  → All their chips are placed in TRAIN only.\n"
            f"  → They will still be valid prediction targets at inference time,\n"
            f"    but val/test metrics will not cover these classes.\n"
            f"  → Add more data for these classes to enable full evaluation."
        )

    rare_rows    = [r for r in rows if r["label"] in rare_classes]
    regular_rows = [r for r in rows if r["label"] not in rare_classes]
    regular_labels = [r["label"] for r in regular_rows]

    if not regular_rows:
        raise ValueError(
            "All classes are too rare to split. Add more data before running this script."
        )

    # ------------------------------------------------------------------ #
    # Stratified split on regular classes only                            #
    # First pull out test+val together, then split those two apart.       #
    # ------------------------------------------------------------------ #
    train_rows, temp_rows, train_labels_split, temp_labels = train_test_split(
        regular_rows, regular_labels,
        test_size=val_test_frac,
        stratify=regular_labels,
        random_state=args.seed,
    )
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=args.test / val_test_frac,
        stratify=temp_labels,
        random_state=args.seed,
    )

    # Merge rare-class rows into train
    train_rows = train_rows + rare_rows

    # ------------------------------------------------------------------ #
    # Write split CSVs                                                     #
    # ------------------------------------------------------------------ #
    splits_dir = args.dataset_dir / "splits"
    write_csv(train_rows, splits_dir / "train.csv")
    write_csv(val_rows,   splits_dir / "val.csv")
    write_csv(test_rows,  splits_dir / "test.csv")

    split_counts = {
        "train": len(train_rows),
        "val":   len(val_rows),
        "test":  len(test_rows),
    }
    print(f"\nSplit sizes  →  train: {split_counts['train']}  |  "
          f"val: {split_counts['val']}  |  test: {split_counts['test']}")

    # Per-class breakdown
    for split_name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        counts = Counter(r["label"] for r in split_rows)
        print(f"  {split_name}: " + "  ".join(f"{c}={counts.get(c, 0)}" for c in classes))

    # ------------------------------------------------------------------ #
    # Write dataset_info.json (consumed by train.py)                      #
    # ------------------------------------------------------------------ #
    info = {
        "input_size": input_size,
        "native_chip_size": native_size,
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "rare_classes": sorted(rare_classes),
        "split_counts": split_counts,
        "split_fractions": {
            "train": args.train,
            "val":   args.val,
            "test":  args.test,
        },
        "seed": args.seed,
    }
    info_path = args.dataset_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nSplit CSVs  →  {splits_dir}/")
    print(f"Dataset info  →  {info_path}")
    print(f"  input_size={input_size}  classes={classes}")


if __name__ == "__main__":
    main()

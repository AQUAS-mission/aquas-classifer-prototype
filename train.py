#!/usr/bin/env python3
"""
Phase 3 — Unified training script for algae species classification.

Supports: mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | efficientnet_b1

Training strategy (two-phase fine-tuning):
  Phase 1 — Frozen backbone: only the classifier head is trained for a few epochs
             to warm it up without destroying pretrained features.
  Phase 2 — Full fine-tune: all layers unlocked, trained with a lower LR and
             cosine annealing schedule.

Outputs (all written to runs/<model>_<timestamp>/):
  best.pt          — best checkpoint (by val accuracy), PyTorch state dict
  last.pt          — final epoch checkpoint
  model.onnx       — ONNX export of the best checkpoint
  metrics.json     — accuracy, per-class precision/recall/F1, confusion matrix

Usage:
    python train.py --model efficientnet_b0
    python train.py --model mobilenet_v3_small --epochs_frozen 5 --epochs_finetune 30
    python train.py --model mobilenet_v3_large --dataset_dir dataset --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    mobilenet_v3_small,
    mobilenet_v3_large,
    efficientnet_b0,
    efficientnet_b1,
)
from PIL import Image
import csv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Supported models registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, tuple] = {
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
    "efficientnet_b0":    (efficientnet_b0,    EfficientNet_B0_Weights.DEFAULT),
    "efficientnet_b1":    (efficientnet_b1,    EfficientNet_B1_Weights.DEFAULT),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight algae classifier with two-phase fine-tuning."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="efficientnet_b0",
        help="Model architecture to train. Default: efficientnet_b0",
    )
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("dataset"),
        help="Root dataset directory (must contain dataset_info.json and splits/). Default: dataset",
    )
    parser.add_argument(
        "--runs_dir", type=Path, default=Path("runs"),
        help="Parent directory for output runs. Default: runs",
    )
    parser.add_argument(
        "--epochs_frozen", type=int, default=5,
        help="Epochs to train with backbone frozen (head warm-up). Default: 5",
    )
    parser.add_argument(
        "--epochs_finetune", type=int, default=15,
        help="Epochs to train with all layers unfrozen. Default: 15",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for train/val. Default: 32",
    )
    parser.add_argument(
        "--lr_head", type=float, default=1e-3,
        help="Learning rate for phase 1 (head only). Default: 1e-3",
    )
    parser.add_argument(
        "--lr_finetune", type=float, default=1e-4,
        help="Learning rate for phase 2 (full fine-tune). Default: 1e-4",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="AdamW weight decay. Default: 1e-4",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing for cross-entropy loss. Default: 0.1",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes. Use 0 on Windows. Default: 0",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def read_split_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ChipDataset(Dataset):
    """Loads individual chip PNGs referenced in a split CSV."""

    def __init__(
        self,
        rows: list[dict],
        dataset_dir: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.dataset_dir = dataset_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        img_path = self.dataset_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[row["label"]]
        return self.transform(image), label


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Load a pretrained model and replace the classifier head to match num_classes.
    All backbone parameters are initially frozen; call unfreeze_backbone() for
    phase 2.
    """
    factory, weights = MODEL_REGISTRY[model_name]
    model = factory(weights=weights)

    # Replace the classifier head
    if model_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        # classifier: Sequential([Linear→Hardswish→Dropout→Linear])
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name in ("efficientnet_b0", "efficientnet_b1"):
        # classifier: Sequential([Dropout, Linear])
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    # Freeze backbone
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    return model


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def classifier_parameters(model: nn.Module) -> list:
    return [p for n, p in model.named_parameters() if n.startswith("classifier")]


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None, runs in eval mode."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(
    model: nn.Module, input_size: int, out_path: Path, device: torch.device
) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------
def build_metrics(
    all_labels: list[int],
    all_preds: list[int],
    classes: list[str],
    run_dir: Path,
) -> dict:
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))

    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall":    report[cls]["recall"],
            "f1":        report[cls]["f1-score"],
            "support":   report[cls]["support"],
        }
        for cls in classes
        if cls in report
    }

    metrics = {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Pretty-print confusion matrix to console
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = f"{'':>16s} " + " ".join(f"{c[:8]:>10s}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = f"{cls[:16]:>16s} " + " ".join(f"{cm[i, j]:>10d}" for j in range(len(classes)))
        print(row)

    print(f"\nTest accuracy: {acc:.4f}")
    print("\nPer-class metrics:")
    for cls, m in per_class.items():
        print(f"  {cls:<20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={int(m['support'])}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Load dataset metadata                                                #
    # ------------------------------------------------------------------ #
    info_path = args.dataset_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"dataset_info.json not found at {info_path}. Run build_dataset.py first."
        )
    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)

    input_size   = info["input_size"]
    classes      = info["classes"]
    class_to_idx = info["class_to_idx"]
    rare_classes = info.get("rare_classes", [])
    num_classes  = info["num_classes"]

    print(f"Classes ({num_classes}): {classes}")
    if rare_classes:
        print(f"Rare classes (train-only): {rare_classes}")
    print(f"Input size: {input_size}x{input_size}")

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #
    # Augmentation for microscopy: organisms can appear at any orientation,
    # so full 360° rotation is valid. Colour jitter handles lighting variation.
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ------------------------------------------------------------------ #
    # Datasets & loaders                                                   #
    # ------------------------------------------------------------------ #
    splits_dir = args.dataset_dir / "splits"

    train_rows = read_split_csv(splits_dir / "train.csv")
    val_rows   = read_split_csv(splits_dir / "val.csv")
    test_rows  = read_split_csv(splits_dir / "test.csv")

    train_ds = ChipDataset(train_rows, args.dataset_dir, class_to_idx, train_transform)
    val_ds   = ChipDataset(val_rows,   args.dataset_dir, class_to_idx, eval_transform)
    test_ds  = ChipDataset(test_rows,  args.dataset_dir, class_to_idx, eval_transform)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\nSplit sizes  →  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    # ------------------------------------------------------------------ #
    # Run directory                                                        #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.runs_dir / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save training config for reproducibility
    config = vars(args).copy()
    config["dataset_dir"] = str(config["dataset_dir"])
    config["runs_dir"]    = str(config["runs_dir"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = build_model(args.model, num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}  |  total params: {total:,}  |  "
          f"trainable (phase 1): {trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ------------------------------------------------------------------ #
    # Phase 1 — head warm-up (backbone frozen)                            #
    # ------------------------------------------------------------------ #
    best_val_acc = 0.0
    best_ckpt_path = run_dir / "best.pt"

    if args.epochs_frozen > 0:
        print(f"\n{'='*60}")
        print(f"Phase 1: head warm-up  ({args.epochs_frozen} epochs, backbone frozen)")
        print(f"{'='*60}")

        optimizer_p1 = AdamW(
            classifier_parameters(model),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
        scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.epochs_frozen, eta_min=1e-6)

        for epoch in range(1, args.epochs_frozen + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p1, device)
            val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,         device)
            scheduler_p1.step()

            elapsed = time.time() - t0
            print(
                f"  [P1] Epoch {epoch:>3}/{args.epochs_frozen}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # ------------------------------------------------------------------ #
    # Phase 2 — full fine-tune (all layers unlocked)                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Phase 2: full fine-tune  ({args.epochs_finetune} epochs, all layers unlocked)")
    print(f"{'='*60}")

    unfreeze_backbone(model)
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (phase 2): {trainable_p2:,}")

    optimizer_p2 = AdamW(
        model.parameters(),
        lr=args.lr_finetune,
        weight_decay=args.weight_decay,
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=args.epochs_finetune, eta_min=1e-6)

    for epoch in range(1, args.epochs_finetune + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,         device)
        scheduler_p2.step()

        elapsed = time.time() - t0
        print(
            f"  [P2] Epoch {epoch:>3}/{args.epochs_finetune}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

    # Save final checkpoint
    last_ckpt_path = run_dir / "last.pt"
    torch.save(model.state_dict(), last_ckpt_path)
    print(f"\nLast checkpoint saved → {last_ckpt_path}")

    # ------------------------------------------------------------------ #
    # Evaluate on test set using best checkpoint                          #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Test evaluation (best checkpoint)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    all_preds, all_labels = collect_predictions(model, test_loader, device)
    metrics = build_metrics(all_labels, all_preds, classes, run_dir)

    # ------------------------------------------------------------------ #
    # ONNX export                                                          #
    # ------------------------------------------------------------------ #
    onnx_path = run_dir / "model.onnx"
    export_onnx(model, input_size, onnx_path, device)
    print(f"\nONNX model exported → {onnx_path}")

    print(f"\nAll outputs in: {run_dir}/")
    print("  best.pt      — best val_acc checkpoint")
    print("  last.pt      — final epoch checkpoint")
    print("  model.onnx   — ONNX export")
    print("  metrics.json — accuracy + per-class metrics + confusion matrix")
    print("  config.json  — training hyperparameters")


if __name__ == "__main__":
    main()

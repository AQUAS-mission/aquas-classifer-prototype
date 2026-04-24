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
try:
    import wandb as _wandb_module
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False
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
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging (requires `pip install wandb` and `wandb login`).",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="algae-classification",
        help="W&B project name. Default: algae-classification",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Resume training from an existing run directory. All hyperparameters "
             "are restored from that run's config.json. Pass --wandb to re-attach "
             "to the original W&B run.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Train all four architectures back-to-back (overrides --model). "
             "Order: efficientnet_b0, efficientnet_b1, mobilenet_v3_small, mobilenet_v3_large.",
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
# Resume checkpoint helpers
# ---------------------------------------------------------------------------

def save_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    phase: int,
    epoch: int,
    best_val_acc: float,
    wandb_run_id: str | None,
    epoch_logs: list[dict],
) -> None:
    """Save full training state so a run can be resumed exactly where it left off."""
    torch.save({
        "phase":           phase,
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_acc":    best_val_acc,
        "wandb_run_id":    wandb_run_id,
        "epoch_logs":      epoch_logs,
    }, path)


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
        opset_version=18,
    )


# ---------------------------------------------------------------------------
# Metrics report + visualizations
# ---------------------------------------------------------------------------

def save_plots(
    metrics: dict,
    run_dir: Path,
    epoch_logs: list[dict],
) -> None:
    """
    Save PNG visualizations to run_dir/plots/:
      confusion_matrix.png   — heatmap
      per_class_f1.png       — horizontal bar chart (F1 per class)
      per_class_prf.png      — grouped bars (P / R / F1 per class)
      training_curves.png    — loss & accuracy over epochs (both phases)
    """
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    classes = metrics["classes"]
    n = len(classes)
    cm = np.array(metrics["confusion_matrix"])
    per_class = metrics["per_class"]

    # ---- 1. Confusion matrix ------------------------------------------ #
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), max(8, n * 0.4)))
    if _SNS_AVAILABLE:
        sns.heatmap(
            cm, annot=(n <= 20), fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes,
            linewidths=0.3 if n <= 20 else 0,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, aspect="auto", cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"Confusion Matrix  (test acc={metrics['accuracy']:.4f})", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    plt.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- 2. Per-class F1 bar chart ------------------------------------ #
    cls_names = list(per_class.keys())
    f1_vals   = [per_class[c]["f1"]        for c in cls_names]
    fig, ax = plt.subplots(figsize=(7, max(5, n * 0.28)))
    colors = ["#d7191c" if v < 0.80 else "#fdae61" if v < 0.90 else "#1a9641" for v in f1_vals]
    ax.barh(cls_names, f1_vals, color=colors)
    ax.axvline(x=metrics["macro_avg"].get("f1-score", 0), color="navy",
               linestyle="--", linewidth=1, label="macro avg")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 score")
    ax.set_title("Per-class F1")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    fig.savefig(plots_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)

    # ---- 3. Grouped P / R / F1 bars ----------------------------------- #
    prec = [per_class[c]["precision"] for c in cls_names]
    rec  = [per_class[c]["recall"]    for c in cls_names]
    x    = np.arange(n)
    w    = 0.26
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 5))
    ax.bar(x - w, prec,    w, label="Precision", color="#4575b4")
    ax.bar(x,     rec,     w, label="Recall",    color="#74add1")
    ax.bar(x + w, f1_vals, w, label="F1",        color="#abd9e9")
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "per_class_prf.png", dpi=150)
    plt.close(fig)

    # ---- 4. Training curves ------------------------------------------- #
    if epoch_logs:
        p1 = [e for e in epoch_logs if e["phase"] == 1]
        p2 = [e for e in epoch_logs if e["phase"] == 2]

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, title in [
            (ax_loss, "loss", "Loss"),
            (ax_acc,  "acc",  "Accuracy"),
        ]:
            offset = 0
            for phase_logs, label_suffix, color_train, color_val in [
                (p1, "(frozen)",   "#e66101", "#fdb863"),
                (p2, "(finetune)", "#5e3c99", "#b2abd2"),
            ]:
                if not phase_logs:
                    continue
                xs = [offset + i + 1 for i in range(len(phase_logs))]
                ax.plot(xs, [e[f"train_{key}"] for e in phase_logs],
                        color=color_train, label=f"train {label_suffix}")
                ax.plot(xs, [e[f"val_{key}"]   for e in phase_logs],
                        color=color_val,   label=f"val {label_suffix}", linestyle="--")
                offset += len(phase_logs)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(fontsize=7)
            if key == "acc":
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        plt.suptitle("Training curves", fontsize=12)
        plt.tight_layout()
        fig.savefig(plots_dir / "training_curves.png", dpi=150)
        plt.close(fig)

    print(f"Plots saved → {plots_dir}/")


def build_metrics(
    all_labels: list[int],
    all_preds: list[int],
    classes: list[str],
    run_dir: Path,
    epoch_logs: list[dict],
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

    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Macro F1:      {metrics['macro_avg'].get('f1-score', 0):.4f}")
    print("\nPer-class metrics:")
    for cls, m in per_class.items():
        print(f"  {cls:<20s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={int(m['support'])}")

    save_plots(metrics, run_dir, epoch_logs)

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
    # Resume state loading                                                 #
    # ------------------------------------------------------------------ #
    resume_state: dict | None = None
    if args.resume is not None:
        resume_dir = args.resume
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        resume_ckpt_path = resume_dir / "resume.pt"
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(
                f"No resume.pt found in {resume_dir}.\n"
                "Only runs started with this version of train.py can be resumed."
            )
        resume_state = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        print(
            f"Resuming from : {resume_dir}\n"
            f"  Completed   : phase {resume_state['phase']}, "
            f"epoch {resume_state['epoch']}\n"
            f"  Best val_acc: {resume_state['best_val_acc']:.4f}"
        )
        # Restore hyperparameters from the original run's config
        saved_cfg = json.loads((resume_dir / "config.json").read_text(encoding="utf-8"))
        for key in (
            "model", "epochs_frozen", "epochs_finetune", "batch_size",
            "lr_head", "lr_finetune", "weight_decay", "label_smoothing",
            "num_workers", "seed",
        ):
            if key in saved_cfg:
                setattr(args, key, saved_cfg[key])

    # ------------------------------------------------------------------ #
    # Run directory                                                        #
    # ------------------------------------------------------------------ #
    if args.resume is not None:
        run_dir = args.resume
        print(f"Run directory: {run_dir}  (resumed)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.runs_dir / f"{args.model}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory: {run_dir}")
        # Save training config for reproducibility
        config = vars(args).copy()
        config["dataset_dir"] = str(config["dataset_dir"])
        config["runs_dir"]    = str(config["runs_dir"])
        config.pop("resume", None)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    resume_ckpt_path = run_dir / "resume.pt"

    # ------------------------------------------------------------------ #
    # W&B initialisation                                                   #
    # ------------------------------------------------------------------ #
    wb = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Run: pip install wandb\n"
                "Then authenticate once with: wandb login"
            )
        wb = _wandb_module
        wb_kwargs: dict = dict(
            project=args.wandb_project,
            config={
                "model":           args.model,
                "epochs_frozen":   args.epochs_frozen,
                "epochs_finetune": args.epochs_finetune,
                "batch_size":      args.batch_size,
                "lr_head":         args.lr_head,
                "lr_finetune":     args.lr_finetune,
                "weight_decay":    args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "input_size":      input_size,
                "num_classes":     num_classes,
                "seed":            args.seed,
                "train_samples":   len(train_ds),
                "val_samples":     len(val_ds),
                "test_samples":    len(test_ds),
            },
            dir=str(run_dir),
        )
        # Re-attach to the original W&B run when resuming
        prior_run_id = (resume_state or {}).get("wandb_run_id")
        if prior_run_id:
            wb_kwargs["resume"] = "must"
            wb_kwargs["id"]     = prior_run_id
        else:
            wb_kwargs["name"] = args.wandb_run_name or run_dir.name
        wb.init(**wb_kwargs)
        print(f"W&B run: {wb.run.url}")

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = build_model(args.model, num_classes).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}  |  total params: {total:,}  |  "
          f"trainable (phase 1): {trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_ckpt_path = run_dir / "best.pt"

    # Restore model weights and best_val_acc when resuming
    best_val_acc = 0.0
    epoch_logs: list[dict] = []  # accumulates {phase, train_loss, train_acc, val_loss, val_acc}
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        best_val_acc = resume_state["best_val_acc"]
        epoch_logs   = resume_state.get("epoch_logs", [])
        print(f"Restored model weights  (best_val_acc so far: {best_val_acc:.4f})")

    # ------------------------------------------------------------------ #
    # Phase 1 — head warm-up (backbone frozen)                            #
    # ------------------------------------------------------------------ #
    # Determine which phase 1 epoch to start from (0 = skip phase entirely)
    p1_start = 1
    if resume_state is not None:
        if resume_state["phase"] == 1:
            p1_start = resume_state["epoch"] + 1
        else:  # phase 2 or beyond — phase 1 already done
            p1_start = args.epochs_frozen + 1

    if args.epochs_frozen > 0 and p1_start <= args.epochs_frozen:
        print(f"\n{'='*60}")
        print(f"Phase 1: head warm-up  ({args.epochs_frozen} epochs, backbone frozen)")
        if p1_start > 1:
            print(f"  Resuming from epoch {p1_start}")
        print(f"{'='*60}")

        optimizer_p1 = AdamW(
            classifier_parameters(model),
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
        scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.epochs_frozen, eta_min=1e-6)

        if resume_state is not None and resume_state["phase"] == 1:
            optimizer_p1.load_state_dict(resume_state["optimizer_state"])
            scheduler_p1.load_state_dict(resume_state["scheduler_state"])

        for epoch in range(p1_start, args.epochs_frozen + 1):
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

            if wb:
                wb.log({
                    "phase": 1,
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc":  train_acc,
                    "val/loss":   val_loss,
                    "val/acc":    val_acc,
                })

            epoch_logs.append({"phase": 1, "train_loss": train_loss, "train_acc": train_acc,
                                "val_loss": val_loss, "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

            save_resume_checkpoint(
                resume_ckpt_path, model, optimizer_p1, scheduler_p1,
                phase=1, epoch=epoch, best_val_acc=best_val_acc,
                wandb_run_id=wb.run.id if wb else None,
                epoch_logs=epoch_logs,
            )

    elif args.epochs_frozen > 0:
        print(f"Phase 1: skipped (already completed in resumed run).")

    # ------------------------------------------------------------------ #
    # Phase 2 — full fine-tune (all layers unlocked)                      #
    # ------------------------------------------------------------------ #
    p2_start = 1
    if resume_state is not None and resume_state["phase"] == 2:
        p2_start = resume_state["epoch"] + 1

    print(f"\n{'='*60}")
    print(f"Phase 2: full fine-tune  ({args.epochs_finetune} epochs, all layers unlocked)")
    if p2_start > 1:
        print(f"  Resuming from epoch {p2_start}")
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

    if resume_state is not None and resume_state["phase"] == 2:
        optimizer_p2.load_state_dict(resume_state["optimizer_state"])
        scheduler_p2.load_state_dict(resume_state["scheduler_state"])

    if p2_start > args.epochs_finetune:
        print("Phase 2: already completed — proceeding to evaluation.")
    else:
        for epoch in range(p2_start, args.epochs_finetune + 1):
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

            if wb:
                wb.log({
                    "phase": 2,
                    "epoch": args.epochs_frozen + epoch,
                    "train/loss": train_loss,
                    "train/acc":  train_acc,
                    "val/loss":   val_loss,
                    "val/acc":    val_acc,
                })

            epoch_logs.append({"phase": 2, "train_loss": train_loss, "train_acc": train_acc,
                                "val_loss": val_loss, "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"    ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")

            save_resume_checkpoint(
                resume_ckpt_path, model, optimizer_p2, scheduler_p2,
                phase=2, epoch=epoch, best_val_acc=best_val_acc,
                wandb_run_id=wb.run.id if wb else None,
                epoch_logs=epoch_logs,
            )

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
    metrics = build_metrics(all_labels, all_preds, classes, run_dir, epoch_logs)

    if wb:
        # Summary scalars
        wb.summary["best_val_acc"] = best_val_acc
        wb.summary["test_acc"]     = metrics["accuracy"]
        wb.summary["macro_f1"]     = metrics["macro_avg"].get("f1-score", 0.0)

        # Per-class metrics table
        table = wb.Table(columns=["class", "precision", "recall", "f1", "support"])
        for cls, m in metrics["per_class"].items():
            table.add_data(cls, round(m["precision"], 4), round(m["recall"], 4),
                           round(m["f1"], 4), int(m["support"]))
        wb.log({"test/per_class_metrics": table})

        # Confusion matrix
        cm = metrics["confusion_matrix"]
        wb.log({
            "test/confusion_matrix": wb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=classes,
            )
        })

        # Upload best checkpoint as artifact
        artifact = wb.Artifact(name=f"{args.model}-best", type="model")
        artifact.add_file(str(best_ckpt_path))
        wb.log_artifact(artifact)

    # ------------------------------------------------------------------ #
    # ONNX export                                                          #
    # ------------------------------------------------------------------ #
    onnx_path = run_dir / "model.onnx"
    try:
        export_onnx(model, input_size, onnx_path, device)
        print(f"\nONNX model exported → {onnx_path}")
    except Exception as onnx_err:
        print(f"\nONNX export skipped: {onnx_err}")
        print("  Install missing dependencies with: pip install onnxscript onnx")

    if wb:
        wb.finish()

    print(f"\nAll outputs in: {run_dir}/")
    print("  best.pt      — best val_acc checkpoint")
    print("  last.pt      — final epoch checkpoint")
    print("  model.onnx   — ONNX export")
    print("  metrics.json — accuracy + per-class metrics + confusion matrix")
    print("  config.json  — training hyperparameters")


if __name__ == "__main__":
    _args = parse_args()
    if _args.all_models:
        _models = list(MODEL_REGISTRY.keys())  # mv3_small, mv3_large, b0, b1
        print(f"Training all {len(_models)} models in sequence: {_models}")
        import sys
        # Snapshot argv once so each iteration starts from the original flags
        _orig_argv = sys.argv[:]
        for _m in _models:
            print(f"\n{'#'*60}")
            print(f"# Starting: {_m}")
            print(f"{'#'*60}\n")
            # Strip --all_models and any existing --model <value> from original argv
            _filtered = []
            _skip = False
            for _a in _orig_argv[1:]:
                if _skip:
                    _skip = False
                    continue
                if _a == "--all_models":
                    continue
                if _a == "--model":
                    _skip = True
                    continue
                _filtered.append(_a)
            sys.argv = [_orig_argv[0], "--model", _m] + _filtered
            main()
    else:
        main()

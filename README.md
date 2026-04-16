# AQUAS Classifier Prototype

A full pipeline for classifying freshwater microalgae species from FlowCam collage images. The pipeline extracts individual particle chips from montage-style `.tif` files, builds a labelled dataset, trains lightweight CNN classifiers, and exports models for edge inference.

## Project Structure

```
raw/labelled/                 # Source labelled library TIFs (multiple layouts)
  10X Libraries/              #   Layout 1: one sub-folder per species
  FlowCam Cyano Example/      #   Layout 2: flat filenames (Anabaena_lib_…tif)
  Libraries - Freshwater Organisms/  # Layout 3: Example_<Species>_10X_… filenames
test_data/                    # Small test TIFs for development
dataset/
  images/<species>/           # Extracted chips (PNG, 600×600px)
  manifest.csv                # Chip index with label, bbox, source path
  dataset_info.json           # Classes, input size, split metadata
  splits/
    train.csv / val.csv / test.csv
runs/<model>_<timestamp>/     # Training outputs per model run
  best.pt / last.pt           # PyTorch checkpoints
  model.onnx                  # ONNX export (opset 17)
  metrics.json                # Accuracy, per-class P/R/F1, confusion matrix
  config.json                 # Hyperparameters used for the run
```

## Dataset

**33 species classes · 7,241 chips · 600×600px (square, black-padded)**

Split: **70% train · 15% val · 15% test** (stratified by class, seed 42)  
Counts: **5,070 train / 1,085 val / 1,086 test**

| Class | Chips | | Class | Chips |
|---|---|---|---|---|
| Anabaena | 643 | | Oscillatoria | 228 |
| Ankistrodesmus | 192 | | Pediastrum | 206 |
| Aphanizomenon | 76 | | Phormidium | 120 |
| Asterionella | 247 | | Planktothrix | 159 |
| Aulacoseira | 13 | | Rotifers | 5 ⚠ |
| Botryococcus | 74 | | Scenedesmus | 317 |
| Ceratium | 70 | | Staurastrum | 7 |
| Chlamydomonas | 76 | | Stephanodiscus | 148 |
| Closterium | 296 | | Synedra | 23 |
| Cosmarium-ish | 8 | | Synura | 25 |
| Cryptomonas | 868 | | Tabellaria | 18 |
| Cyclotella | 173 | | Uroglenopsis | 340 |
| Cylindrospermopsis | 93 | | Volvox | 282 |
| Dinobryon | 42 | | | |
| Euglena | 528 | | | |
| Fragilaria | 99 | | | |
| LRGT | 10 | | | |
| Lyngbya | 44 | | | |
| Mallomonas | 1,606 | | | |
| Microcystis | 205 | | | |

⚠ Rare class — too few chips for val/test splits; placed in train only until more data is added.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Pipeline

### Phase 1 — Extract chips from library TIFs

```bash
# Process all labelled library folders under raw/labelled/ into dataset/
python process_labelled.py

# Or extract chips from a flat folder of TIFs (e.g. test_data/)
python tif_montage_to_dataset.py --input_dir test_data --output_dir dataset
```

`process_labelled.py` automatically detects three source layouts:

| Layout | Example path | Label derived from |
|---|---|---|
| Species-directory | `10X Libraries/Anabaena/*.tif` | Parent directory name |
| Flat filename | `FlowCam Cyano Example/Libraries/Anabaena_lib_…tif` | Filename stem before first `_` |
| Example-prefix | `Libraries - Freshwater Organisms/…/Example_Anabaena_10X_…tif` | Strip `Example_` prefix, take part before `_10X` |

New chips are always appended — existing chips are never overwritten. If a new source image is larger than the current chip size (up to the 600px cap), existing chips are automatically re-padded.

### Phase 2 — Build train/val/test splits

```bash
python build_dataset.py
```

Reads `dataset/manifest.csv`, performs a stratified split, and writes
`dataset/splits/train.csv`, `val.csv`, `test.csv`, and updates `dataset/dataset_info.json`.

### Phase 3 — Train a model

```bash
python train.py --model efficientnet_b0
```

Available models:

| `--model` | Params | ONNX size | Notes |
|---|---|---|---|
| `mobilenet_v3_small` | 2.5M | ~10MB | Smallest / fastest |
| `mobilenet_v3_large` | 5.5M | ~21MB | Mobile sweet spot |
| `efficientnet_b0` | 5.3M | ~20MB | Best accuracy/size — recommended |
| `efficientnet_b1` | 7.8M | ~30MB | Accuracy ceiling |

Key options:

| Option | Default | Description |
|---|---|---|
| `--epochs_frozen` | `5` | Head warm-up epochs (backbone frozen) |
| `--epochs_finetune` | `30` | Full fine-tune epochs |
| `--batch_size` | `32` | Training batch size |
| `--lr_head` | `1e-3` | Phase 1 learning rate |
| `--lr_finetune` | `1e-4` | Phase 2 learning rate |

Training is two-phase: the backbone is frozen for `--epochs_frozen` epochs to warm up the classifier head, then all weights are unfrozen for full fine-tuning. Cosine annealing is used for both phases. Each run is saved to `runs/<model>_<timestamp>/`.

## Script Reference

| Script | Purpose |
|---|---|
| `tif_montage_to_dataset.py` | Extract chips from a flat folder of TIFs; derives label from filename stem |
| `process_labelled.py` | Extract chips from all sub-folders under `raw/labelled/`; handles 3 source layouts; supports incremental updates |
| `build_dataset.py` | Stratified 70/15/15 train/val/test split; outputs split CSVs and `dataset_info.json` |
| `train.py` | Unified two-phase training for all 4 model architectures; exports ONNX (opset 17) + metrics |

## Image Size

All chips are stored at **600×600px** (RGB, black-padded to square). Chips extracted from sources smaller than 600px are padded up; chips from larger sources are scaled down with Lanczos resampling before padding. The cap is enforced in both `tif_montage_to_dataset.py` and `process_labelled.py`.

## Adding More Data

1. Add new labelled TIF files to the appropriate sub-folder under `raw/labelled/`  
   (or create a new `<SpeciesName>/` folder under `raw/labelled/10X Libraries/`)
2. Run `python process_labelled.py` — new chips are appended without touching existing ones
3. Run `python build_dataset.py` — splits are regenerated
4. Re-run `python train.py --model <model>` to retrain

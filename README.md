# Algae Classification Dataset Builder

Converts montage-style `.tif` files (one per algae species) into individual images compatible with standard ML image classification pipelines.

## Format

Each `.tif` in `test_data/` contains many specimen images arranged in rows on a black background. The script detects row and column boundaries via black pixel strips and extracts each specimen as its own image.

**Class name** is derived from the filename — everything before the first `_`.  
Example: `Anabaena_lib_images_000001.tif` → class `Anabaena`

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

`MAX_SIZE` is a global variable at the top of `tif_montage_to_dataset.py`. If an image is smaller than `MAX_SIZE` in either dimension, black pixels are added on the right and/or bottom. Images larger than `MAX_SIZE` are kept at their natural size. Set to `None` to skip padding.

```python
MAX_SIZE: int | None = 128
```

## Usage

```bash
source venv/bin/activate
python tif_montage_to_dataset.py [options]
```

| Option | Default | Description |
|---|---|---|
| `--input_dir` | `test_data` | Directory containing `.tif`/`.tiff` files |
| `--output_dir` | `output_data` | Directory to write individual images |
| `--black_threshold` | `10` | Max channel value to treat a pixel as black background |

### Examples

```bash
python tif_montage_to_dataset.py
python tif_montage_to_dataset.py --input_dir raw_tifs --output_dir dataset
```

## Output

Images are saved to `output_data/` as `<classname>_<number>.png`, e.g.:

```
output_data/
  Anabaena_1.png
  Anabaena_2.png
  ...
  Asterionella_1.png
  ...
```

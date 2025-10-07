# Segmentation Module (segment)

This directory organizes segmentation-related code. It keeps training/testing and data preprocessing (including MedNeXt). Visualization/inference scripts are excluded.

## Structure

```
segment/
  README.md
  weights/                 # Model weights (recommend .gitignore)
  dataset/                 # (Optional) dataset helpers/configs
  models/                  # Reusable models and tooling
    utils.py               # Shared losses/utilities (from project root utils.py)
    mednextv1/             # MedNeXt implementation
      __init__.py
      blocks.py
      create_mednext_v1.py
      MedNextV1.py
  unet/                    # U-Net train/test
    train_unet.py
    test_unet.py
  mednext/                 # MedNeXt train/test
    train_mednext.py
    test_mednext.py
  preprocess/              # Data preprocessing
    data_preprocess.py     # Min bounding-rect crop + fixed-size resize
```

## Environment

```bash
conda activate OCT
```

## Run (examples)

```bash
# U-Net training
python segment/unet/train_unet.py --data_dir <path_to_data> --save_dir segment/weights

# U-Net evaluation
python segment/unet/test_unet.py --data_dir <path_to_val> --weights_dir segment/weights

# MedNeXt training
python segment/mednext/train_mednext.py

# MedNeXt evaluation
python segment/mednext/test_mednext.py --weights_dir segment/weights
```

> If scripts import modules from the project root (e.g., `from utils import ...`), run from the repository root or set `PYTHONPATH`: `export PYTHONPATH="$PWD:$PYTHONPATH"`.

## .gitignore suggestions

```
segment/weights/*.pth
segment/visualization/**
```

## Notes

- Visualization/inference scripts and outputs are not included here.
- If you need a minimal utilities set, trim unrelated parts in `models/utils.py`.

## Preprocessing (example)

Currently only the "min bounding-rect crop + fixed-size resize" preprocessing is provided.

```bash
# Run directly (uses defaults)
python segment/preprocess/data_preprocess.py

# Optional arguments (all have defaults)
python segment/preprocess/data_preprocess.py \
  --mask_root "data_lung_ultrasound/mask" \
  --image_root "data_lung_ultrasound/image" \
  --output_mask_root "data_lung_ultrasound_cropped/mask" \
  --output_image_root "data_lung_ultrasound_cropped/image" \
  --target_size 256
```

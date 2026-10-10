# ArroyoWaste Density Counting

Code and frozen manifests for the ArroyoWaste and ABD density-counting experiments. Run every command from the repository root.

## Environment

```bash
conda env create -f environment.yml
conda activate geomcu
```

## Data

### ArroyoWaste

Open the [ArroyoWaste data package](https://doi.org/10.6073/pasta/f29890eebef9b36bb128d5165cbdd433) and download only:

1. ArroyoWaste v1.0 image collection.
2. ArroyoWaste v1.0 annotations.

Extract both archives into `data/arroyowaste/`. The result must contain:

```text
data/arroyowaste/
├── images/
└── annotations/
    └── points/
```

Verify the files:

```bash
python scripts/verify_data.py \
  --root data/arroyowaste \
  --manifest manifests/arroyowaste.csv
```

The expected result is 414 samples and 23,235 points.

### ABD

ABD adds 398 BePLi samples and 640 DSWD samples to ArroyoWaste.

Download and extract:

- [BePLi Dataset v2](https://www.seanoe.org/data/00858/96963/): `106157.tar.gz`.
- [DSWD](https://doi.org/10.17632/gr99ny6b8p.1): `DSWD.zip`.

After extraction, arrange the required raw files as follows. An extra top-level folder created by an archive may be discarded.

```text
data/raw/
├── bepli/
│   └── plastic_coco/
│       ├── annotation/
│       │   ├── train.json
│       │   ├── val.json
│       │   └── test.json
│       └── images/
│           └── original_images/    # 3,722 .png images
└── dswd/
    └── Train/
        ├── Image/                  # 640 img_*.png files
        └── Mask/                   # 640 mask_*.png files
```

For BePLi, use `original_images`, not the separate `images/train`, `images/val`, or `images/test` folders. For DSWD, only the `Train` split is used.

Prepare both sources:

```bash
python scripts/prepare_bepli.py \
  --images data/raw/bepli/plastic_coco/images/original_images \
  --annotations data/raw/bepli/plastic_coco/annotation \
  --output data/sources/bepli/train_data

python scripts/prepare_dswd.py \
  --images data/raw/dswd/Train/Image \
  --masks data/raw/dswd/Train/Mask \
  --output data/sources/dswd/train_data
```

The scripts create:

```text
data/sources/
├── bepli/train_data/
│   ├── images/         # 398 images
│   └── annotations/    # 398 .npy files
└── dswd/train_data/
    ├── images/         # 640 images
    └── annotations/    # 640 .npy files
```

Build and verify ABD:

```bash
python scripts/build_abd.py
python scripts/verify_data.py --root data/abd --manifest manifests/abd.csv
```

The expected result is 1,452 samples and 36,801 points.

## Density caches

```bash
python scripts/build_density_cache.py \
  --dataset arroyowaste \
  --manifest manifests/arroyowaste.csv

python scripts/build_density_cache.py \
  --dataset abd \
  --manifest manifests/abd.csv
```

Both commands use the frozen recipe `adaptive_b03_k3_s40_clip10-120_stride4_native`.

## Train, evaluate, and visualize

Choose one of the configurations under `configs/arroyowaste/` or `configs/abd/`.

```bash
python src/train.py configs/arroyowaste/seed7.json

python src/evaluate.py \
  configs/arroyowaste/seed7.json \
  path/to/best_game2.pt

python src/visualize.py \
  configs/arroyowaste/seed7.json \
  path/to/best_game2.pt
```

The MCNN contains 85,057 trainable parameters. Frozen reference metrics are available in `results/`.

| Training data | MAE | NAE | nGAME(2) | Sinkhorn |
|---|---:|---:|---:|---:|
| ArroyoWaste | 52.54 ± 1.84 | 0.641 ± 0.022 | 0.886 ± 0.009 | 0.0947 ± 0.0022 |
| ABD | 48.15 ± 1.36 | 0.588 ± 0.017 | 0.838 ± 0.014 | 0.0858 ± 0.0035 |

The frozen ABD reconstruction previously matched all 1,452 image and point hashes. Its 1,452 generated density arrays also matched the experimental cache exactly, with maximum absolute difference 0.0.

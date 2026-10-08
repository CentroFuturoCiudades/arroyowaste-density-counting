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

ABD adds 398 BePLi samples and 640 DWSD samples to ArroyoWaste.

- [BePLi Dataset v1](https://doi.org/10.17882/92297): download `98753.zip`.
- [DWSD](https://doi.org/10.17632/gr99ny6b8p.1): download `DSWD.zip` and use its 640-image training split.

These official archives contain raw COCO annotations and segmentation masks. `build_abd.py` requires the corresponding prepared point annotations:

```text
data/sources/
├── bepli/train_data/
│   ├── images/        # 398 images
│   └── annotations/   # 398 .npy point files
└── dswd/train_data/
    ├── images/        # 640 images
    └── annotations/   # 640 .npy point files
```

The exact raw-to-point conversion is not included in this release. Therefore, ABD cannot yet be reconstructed from the two official raw downloads alone. Do not run the next command until the prepared source folders are available.

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

# ArroyoWaste Density Counting

This repository contains the reproducibility materials for the density-counting experiments associated with the ArroyoWaste dataset. It provides the frozen ArroyoWaste and ABD manifests, the MCNN counting model, the density-target recipe, experiment configurations, training and evaluation programs, visualization tools, and reference metrics.

The image datasets are not redistributed. Download them from their original sources and arrange them as described below. Run all commands from the repository root.

## Installation

Create an isolated environment and install the direct runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch uses CUDA when a compatible installation and device are available. Pass `--device cpu` to the training, evaluation, or visualization command to select the CPU explicitly.

## Dataset preparation

### ArroyoWaste

ArroyoWaste is the primary evaluation dataset. Download it manually from the [Environmental Data Initiative repository](https://doi.org/10.6073/pasta/f29890eebef9b36bb128d5165cbdd433) and place the benchmark files under:

```text
data/arroyowaste/
```

The frozen benchmark contains 414 samples and 23,235 point annotations:

| Split | Samples |
|---|---:|
| train | 392 |
| validation | 11 |
| test | 11 |

Verify the installed files, annotation counts, and SHA-256 hashes against the frozen manifest:

```bash
python scripts/verify_data.py \
  --root data/arroyowaste \
  --manifest manifests/arroyowaste.csv
```

### ABD source datasets

ABD combines ArroyoWaste with two additional training sources:

- BePLi: 398 samples. See the [dataset publication](https://doi.org/10.1016/j.dib.2023.109176).
- Dense Waste Segmentation Dataset (DWSD): 640 samples. See the [dataset publication](https://doi.org/10.1016/j.dib.2025.111340) and [data record](https://doi.org/10.17632/gr99ny6b8p.1).

Download the source datasets and arrange the required subsets as follows:

```text
data/sources/
├── bepli/
│   └── train_data/
│       ├── images/
│       └── annotations/
└── dswd/
    └── train_data/
        ├── images/
        └── annotations/
```

Each annotation file must be a NumPy array named after its corresponding image stem, with shape `(N, 2)` and point coordinates in `(x, y)` order. The frozen manifest determines the exact source samples used.

## ABD reconstruction

ABD is the combined corpus used in the transfer experiment. Its frozen composition is:

| Source | Samples |
|---|---:|
| ArroyoWaste | 414 |
| BePLi | 398 |
| DWSD | 640 |
| **Total** | **1,452** |

The experimental split contains 1,430 training, 11 validation, and 11 test samples. Only ArroyoWaste samples are used for validation and testing; BePLi and DWSD only expand the training split.

After preparing all three sources, reconstruct ABD:

```bash
python scripts/build_abd.py
```

The command creates `data/abd/` and validates every generated image and point file against the frozen SHA-256 hashes in `manifests/abd.csv`. It refuses to overwrite a non-empty output directory.

Verify the reconstructed corpus independently:

```bash
python scripts/verify_data.py \
  --root data/abd \
  --manifest manifests/abd.csv
```

A correct reconstruction contains 1,452 samples and 36,801 point annotations.

## Density-cache generation

The model is trained with geometry-adaptive density targets. The frozen recipe is:

```text
recipe ID       adaptive_b03_k3_s40_clip10-120_stride4_native
beta            0.3
k neighbors     3
fallback sigma  4.0
minimum sigma   1.0
maximum sigma   12.0
output stride   4
```

For 1024 × 768 inputs, density maps have shape 256 × 192. The implementation preserves the total annotation mass after Gaussian generation and downsampling.

Generate each cache with its explicit frozen manifest:

```bash
python scripts/build_density_cache.py \
  --dataset arroyowaste \
  --manifest manifests/arroyowaste.csv

python scripts/build_density_cache.py \
  --dataset abd \
  --manifest manifests/abd.csv
```

Each command writes beneath the corresponding dataset root:

```text
density_maps/adaptive_b03_k3_s40_clip10-120_stride4_native/
```

Existing cache arrays are validated and retained unless `--overwrite` is supplied.

## Model and training

The model in `src/models/mcnn.py` is the exact four-branch MCNN architecture used for the reported experiments and contains **85,057 trainable parameters**.

Six configurations cover the two training corpora and three experimental seeds:

```text
configs/
├── arroyowaste/
│   ├── seed7.json
│   ├── seed17.json
│   └── seed27.json
└── abd/
    ├── seed7.json
    ├── seed17.json
    └── seed27.json
```

Train one configuration at a time:

```bash
python src/train.py configs/arroyowaste/seed7.json
python src/train.py configs/abd/seed7.json
```

Use `seed17.json` and `seed27.json` for the remaining runs. The default protocol is 100 epochs, batch size 6, gradient accumulation over 2 steps, learning rate `1e-3`, minimum learning rate `1e-5`, weight decay `1e-4`, density scale 100, count-loss weight 0.01, and GAME-loss weight 0.5 at levels 1 and 2. Checkpoint selection uses the best validation GAME(2).

## Evaluation

Evaluate a checkpoint on the frozen test split:

```bash
python src/evaluate.py \
  configs/abd/seed7.json \
  path/to/best_game2.pt \
  --output evaluation/abd_seed7
```

The evaluator writes `metrics.json` and `per_image.csv` and reports MAE, NAE, normalized GAME(2), and Sinkhorn divergence. The Sinkhorn metric uses a mass-preserving 48 × 64 representation, epsilon 0.05, and 200 iterations.

## Visualization

Generate qualitative predictions without changing the underlying density outputs:

```bash
python src/visualize.py \
  configs/abd/seed7.json \
  path/to/best_game2.pt \
  --output visualizations/abd_seed7
```

The default selection is three representative test samples. The script visualizes the raw density map produced by the model.

## Reference metrics

The frozen paper results are stored in:

```text
results/
├── paired_comparison.csv
├── reference_metrics.csv
└── reference_metrics_per_seed.csv
```

For the `best_game2` checkpoint, the three-seed aggregate results are:

| Training corpus | MAE | NAE | nGAME(2) | Sinkhorn |
|---|---:|---:|---:|---:|
| ArroyoWaste | 52.54 ± 1.84 | 0.641 ± 0.022 | 0.886 ± 0.009 | 0.0947 ± 0.0022 |
| ABD | 48.15 ± 1.36 | 0.588 ± 0.017 | 0.838 ± 0.014 | 0.0858 ± 0.0035 |

## Repository structure

```text
.
├── configs/                 # Frozen experiment configurations
│   ├── abd/
│   └── arroyowaste/
├── manifests/               # Frozen sample lists and SHA-256 hashes
│   ├── abd.csv
│   └── arroyowaste.csv
├── results/                 # Frozen aggregate and per-seed metrics
├── scripts/
│   ├── build_abd.py         # Reconstruct and hash-check ABD
│   ├── build_density_cache.py
│   └── verify_data.py
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluate.py
│   ├── losses.py
│   ├── spatial_metrics.py
│   ├── train.py
│   └── visualize.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets, caches, training runs, and checkpoints are intentionally excluded from version control.

## Reproducibility statement

The public ABD reconstruction was validated against the frozen experimental corpus. All 1,452 reconstructed images and point files matched their expected SHA-256 hashes exactly. All 1,452 generated density arrays matched the historical experimental cache exactly in shape and value, with a maximum absolute difference of 0.0.

The manifests, density recipe, MCNN architecture, training and loss implementations, evaluation metrics, visualization implementation, and reference results in this repository are treated as frozen research artifacts.

## Citation

If you use ArroyoWaste, cite the associated dataset using its [persistent DOI](https://doi.org/10.6073/pasta/f29890eebef9b36bb128d5165cbdd433). Repository-level software citation metadata will require the release authors and publication details to be supplied by the maintainers.

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import game_metric
from spatial_metrics import spatial_sinkhorn_divergence
from train import make_datasets, make_model, load_checkpoint


@torch.no_grad()
def evaluate(
    cfg,
    checkpoint_path,
    device,
    split="test",
    sinkhorn_size=(48, 64),
    sinkhorn_epsilon=0.05,
    sinkhorn_iters=200,
):
    _, dataset = make_datasets(
        cfg,
        eval_split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(
            cfg["train"].get(
                "num_workers",
                0,
            )
        ),
        pin_memory=bool(
            cfg["train"].get(
                "pin_memory",
                True,
            )
        ),
    )

    model, _ = make_model(cfg)

    load_checkpoint(
        model,
        checkpoint_path,
    )

    model = model.to(device)
    model.eval()

    rows = []

    total_abs_error = 0.0
    total_gt = 0.0
    total_game2 = 0.0
    sinkhorn_values = []

    for index, (images, density) in enumerate(
        tqdm(
            loader,
            desc=f"Evaluating {split}",
        )
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        density = density.to(
            device,
            non_blocking=True,
        )

        pred = model(images)

        pred_count = float(
            pred.sum().detach().cpu()
        )

        gt_count = float(
            density.sum().detach().cpu()
        )

        abs_error = abs(
            pred_count - gt_count
        )

        game2 = float(
            game_metric(
                pred,
                density,
                level=2,
            )
            .detach()
            .cpu()
        )

        sinkhorn = float(
            spatial_sinkhorn_divergence(
                pred,
                density,
                out_hw=sinkhorn_size,
                epsilon=sinkhorn_epsilon,
                n_iters=sinkhorn_iters,
            )
            .detach()
            .cpu()
        )

        total_abs_error += abs_error
        total_gt += gt_count
        total_game2 += game2
        sinkhorn_values.append(sinkhorn)

        rows.append({
            "index": index,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "abs_error": abs_error,
            "game2": game2,
            "sinkhorn": sinkhorn,
        })

    n = len(rows)

    if n == 0:
        raise RuntimeError(
            f"No samples found for split={split!r}"
        )

    metrics = {
        "split": split,
        "n_samples": n,

        "mae":
            total_abs_error
            / n,

        "nae":
            total_abs_error
            / max(
                total_gt,
                1e-12,
            ),

        "ngame2":
            total_game2
            / max(
                total_gt,
                1e-12,
            ),

        "sinkhorn":
            float(
                np.mean(
                    sinkhorn_values
                )
            ),

        "total_gt":
            total_gt,

        "sinkhorn_settings": {
            "height":
                int(
                    sinkhorn_size[0]
                ),

            "width":
                int(
                    sinkhorn_size[1]
                ),

            "epsilon":
                float(
                    sinkhorn_epsilon
                ),

            "iterations":
                int(
                    sinkhorn_iters
                ),
        },
    }

    return metrics, rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=Path,
        help="Experiment JSON config.",
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Checkpoint to evaluate.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--sinkhorn-height",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--sinkhorn-width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--sinkhorn-epsilon",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    cfg = json.loads(
        args.config.read_text()
    )

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(
            args.device
        )

    metrics, rows = evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=device,
        split=args.split,
        sinkhorn_size=(
            args.sinkhorn_height,
            args.sinkhorn_width,
        ),
        sinkhorn_epsilon=
            args.sinkhorn_epsilon,
        sinkhorn_iters=
            args.sinkhorn_iters,
    )

    if args.output is None:
        output_dir = (
            args.checkpoint.parent
            / f"evaluation_{args.split}"
        )
    else:
        output_dir = args.output

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        )
        + "\n"
    )

    with (
        output_dir
        / "per_image.csv"
    ).open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "gt_count",
                "pred_count",
                "abs_error",
                "game2",
                "sinkhorn",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Split: {metrics['split']}")
    print(f"N: {metrics['n_samples']}")
    print(
        f"MAE: {metrics['mae']:.6f}"
    )
    print(
        f"NAE: {metrics['nae']:.6f}"
    )
    print(
        f"nGAME2: {metrics['ngame2']:.6f}"
    )
    print(
        "Sinkhorn: "
        f"{metrics['sinkhorn']:.6f}"
    )

    print(
        f"\nResults: {output_dir}"
    )


if __name__ == "__main__":
    main()

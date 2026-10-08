import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train import (
    load_checkpoint,
    make_datasets,
    make_model,
)


def sample_id(dataset, index):
    if hasattr(dataset, "samples"):
        sample = dataset.samples[index]

        if isinstance(sample, dict):
            value = sample.get(
                "sample_id",
                sample.get(
                    "global_id",
                    index,
                ),
            )

            return str(value)

    if hasattr(dataset, "ids"):
        return str(
            dataset.ids[index]
        )

    return str(index)


def image_to_numpy(image):
    image = (
        image.detach()
        .cpu()
        .float()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    return np.clip(
        image,
        0.0,
        1.0,
    )


@torch.no_grad()
def run_inference(
    model,
    dataset,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    predictions = {}

    model.eval()

    for index, (
        image,
        target,
    ) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        prediction = model(image)

        gt_count = float(
            target.sum().cpu()
        )

        pred_count = float(
            prediction.sum().cpu()
        )

        sid = sample_id(
            dataset,
            index,
        )

        abs_error = abs(
            pred_count
            - gt_count
        )

        rows.append({
            "index": index,
            "id": sid,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "abs_error": abs_error,
        })

        predictions[index] = {
            "image":
                image[0].cpu(),

            "target":
                target[
                    0,
                    0,
                ].cpu().numpy(),

            "prediction":
                prediction[
                    0,
                    0,
                ].cpu().numpy(),
        }

    return (
        pd.DataFrame(rows),
        predictions,
    )


def select_samples(
    dataframe,
    mode,
    n,
):
    dataframe = dataframe.sort_values(
        "abs_error"
    ).reset_index(
        drop=True
    )

    n = min(
        n,
        len(dataframe),
    )

    if mode == "best":
        return dataframe.head(n)

    if mode == "worst":
        return dataframe.tail(n)

    if mode == "representative":
        indices = np.linspace(
            0,
            len(dataframe) - 1,
            n,
        )

        indices = np.round(
            indices
        ).astype(int)

        return dataframe.iloc[
            indices
        ]

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def plot_sample(
    row,
    predictions,
    output_path,
):
    item = predictions[
        int(row["index"])
    ]

    image = image_to_numpy(
        item["image"]
    )

    target = item["target"]
    prediction = item["prediction"]

    vmax = max(
        float(target.max()),
        float(prediction.max()),
        1e-12,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(image)
    axes[0].set_title("Image")

    axes[1].imshow(
        target,
        cmap="jet",
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].set_title(
        f"Ground truth\n"
        f"Count = "
        f"{row['gt_count']:.1f}"
    )

    axes[2].imshow(
        prediction,
        cmap="jet",
        vmin=0.0,
        vmax=vmax,
    )
    axes[2].set_title(
        f"Prediction\n"
        f"Count = "
        f"{row['pred_count']:.1f}"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{row['id']} | "
        f"Absolute error = "
        f"{row['abs_error']:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize MCNN density "
            "predictions."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
    )

    parser.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "best",
            "worst",
            "representative",
        ],
        default="representative",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "visualizations"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
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

    _, dataset = make_datasets(
        cfg,
        eval_split=args.split,
    )

    model, _ = make_model(cfg)

    load_checkpoint(
        model,
        args.checkpoint,
    )

    model = (
        model
        .to(device)
        .eval()
    )

    dataframe, predictions = (
        run_inference(
            model,
            dataset,
            device,
        )
    )

    selected = select_samples(
        dataframe,
        args.mode,
        args.n,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        args.output
        / "selected_samples.csv",
        index=False,
    )

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        output_path = (
            args.output
            / (
                f"{rank:02d}_"
                f"{row['id']}.png"
            )
        )

        plot_sample(
            row,
            predictions,
            output_path,
        )

        print(
            f"{rank}: "
            f"{row['id']} | "
            f"GT={row['gt_count']:.2f} | "
            f"Pred={row['pred_count']:.2f} | "
            f"AE={row['abs_error']:.2f}"
        )

    print(
        f"\nSaved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()

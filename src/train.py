import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.arroyowaste import ArroyoWasteDataset
from data.abd import ABDDataset
from losses import density_count_loss, game_metric
from models.mcnn import MCNN


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# Dataset
# ============================================================

def make_datasets(cfg, eval_split="val"):
    dcfg = cfg["dataset"]

    dataset_mode = dcfg["dataset_mode"]
    root = dcfg["root"]

    cache_cfg = dcfg.get(
        "density_cache",
        {},
    )

    common = {
        "root": root,
        "split_manifest":
            dcfg["split_manifest"],

        "out_stride":
            int(dcfg.get("out_stride", 4)),

        "sigma":
            float(dcfg.get("sigma", 4.0)),

        "adaptive":
            bool(dcfg.get("adaptive", True)),

        "beta":
            float(dcfg.get("beta", 0.3)),

        "k_neighbors":
            int(dcfg.get("k_neighbors", 3)),

        "sigma_min":
            float(dcfg.get("sigma_min", 1.0)),

        "sigma_max":
            float(dcfg.get("sigma_max", 12.0)),

        "density_cache_mode":
            cache_cfg.get(
                "mode",
                "require",
            ),

        "density_cache_root":
            cache_cfg.get("root"),

        "return_metadata": False,
    }

    if dataset_mode == "arroyowaste":
        train_ds = ArroyoWasteDataset(
            split="train",
            **common,
        )

        eval_ds = ArroyoWasteDataset(
            split=eval_split,
            **common,
        )

        return train_ds, eval_ds

    if dataset_mode == "abd":
        common["resize_mode"] = dcfg.get(
            "resize_mode",
            "none",
        )

        train_ds = ABDDataset(
            split="train",
            **common,
        )

        eval_ds = ABDDataset(
            split=eval_split,
            **common,
        )

        return train_ds, eval_ds

    raise ValueError(
        f"Unknown dataset_mode: {dataset_mode!r}. "
        "Use 'arroyowaste' or 'abd'."
    )


# ============================================================
# Model
# ============================================================

def make_model(cfg):
    mcfg = cfg["model"]

    if mcfg.get("name", "mcnn") != "mcnn":
        raise ValueError(
            "This repository supports only model='mcnn'."
        )

    model = MCNN(
        in_channels=mcfg.get(
            "in_channels",
            3,
        ),
        expansion=mcfg.get(
            "expansion",
            3,
        ),
        final_activation=mcfg.get(
            "final_activation",
            "softplus",
        ),
    )

    return model, mcfg


def load_checkpoint(
    model,
    checkpoint_path,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(
        state,
        strict=True,
    )

    return model


# ============================================================
# DataLoaders
# ============================================================

def make_loaders(
    cfg,
    train_ds,
    val_ds,
):
    tcfg = cfg["train"]

    generator = torch.Generator()
    generator.manual_seed(
        int(
            tcfg.get(
                "data_order_seed",
                tcfg["seed"],
            )
        )
    )

    num_workers = int(
        tcfg.get("num_workers", 0)
    )

    common = {
        "num_workers": num_workers,
        "pin_memory": bool(
            tcfg.get(
                "pin_memory",
                True,
            )
        ),
    }

    if num_workers > 0:
        common["persistent_workers"] = bool(
            tcfg.get(
                "persistent_workers",
                True,
            )
        )

        common["prefetch_factor"] = int(
            tcfg.get(
                "prefetch_factor",
                2,
            )
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            tcfg["batch_size"]
        ),
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        **common,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(
            tcfg.get(
                "val_batch_size",
                1,
            )
        ),
        shuffle=False,
        **common,
    )

    return train_loader, val_loader


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    absolute_errors = []
    game2_values = []

    for images, density in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        density = density.to(
            device,
            non_blocking=True,
        )

        pred = model(images)

        pred_count = pred.sum(
            dim=(1, 2, 3)
        )

        true_count = density.sum(
            dim=(1, 2, 3)
        )

        absolute_errors.extend(
            torch.abs(
                pred_count - true_count
            )
            .detach()
            .cpu()
            .tolist()
        )

        # Canonical GAME2 averaged over this batch.
        game2 = game_metric(
            pred,
            density,
            level=2,
        )

        # Validation batch size is 1 in the reported runs.
        game2_values.append(
            float(
                game2.detach().cpu()
            )
        )

    return {
        "mae": float(
            np.mean(absolute_errors)
        ),
        "game2": float(
            np.mean(game2_values)
        ),
    }


# ============================================================
# Training
# ============================================================

def train(
    cfg,
    device,
):
    tcfg = cfg["train"]
    lcfg = cfg["loss"]

    seed = int(tcfg["seed"])

    set_seed(
        seed,
        deterministic=bool(
            tcfg.get(
                "deterministic",
                False,
            )
        ),
    )

    train_ds, val_ds = make_datasets(
        cfg,
        eval_split="val",
    )

    train_loader, val_loader = (
        make_loaders(
            cfg,
            train_ds,
            val_ds,
        )
    )

    model, _ = make_model(cfg)
    model = model.to(device)

    epochs = int(tcfg["epochs"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(
            tcfg["weight_decay"]
        ),
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(
                tcfg["min_lr"]
            ),
        )
    )

    amp_enabled = (
        device.type == "cuda"
        and bool(
            tcfg.get(
                "use_amp",
                True,
            )
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    accumulation_steps = int(
        tcfg.get(
            "gradient_accumulation_steps",
            1,
        )
    )

    eval_every = int(
        tcfg.get(
            "eval_every",
            1,
        )
    )

    run_root = Path(
        cfg.get(
            "runs_dir",
            "runs",
        )
    )

    run_name = cfg.get(
        "run_name",
        f"seed{seed}",
    )

    run_dir = run_root / run_name
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = (
        run_dir
        / "config.json"
    )

    config_path.write_text(
        json.dumps(
            cfg,
            indent=2,
        )
        + "\n"
    )

    history_path = (
        run_dir
        / "history.csv"
    )

    checkpoint_path = (
        run_dir
        / "best_game2.pt"
    )

    with history_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_density_loss",
            "train_count_loss",
            "train_game_loss",
            "val_mae",
            "val_game2",
            "best_game2",
        ])

    best_game2 = float("inf")
    best_epoch = -1

    print(
        f"Device: {device}"
    )
    print(
        f"Train samples: {len(train_ds)}"
    )
    print(
        f"Validation samples: {len(val_ds)}"
    )
    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )
    print(
        f"AMP: {amp_enabled}"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        totals = {
            "loss": 0.0,
            "density": 0.0,
            "count": 0.0,
            "game": 0.0,
        }

        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{epochs}"
            ),
        )

        for batch_idx, (
            images,
            density,
        ) in enumerate(pbar):
            images = images.to(
                device,
                non_blocking=True,
            )

            density = density.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
            ):
                pred = model(images)

                loss, logs = (
                    density_count_loss(
                        pred,
                        density,

                        density_scale=float(
                            lcfg[
                                "density_scale"
                            ]
                        ),

                        lambda_count=float(
                            lcfg[
                                "lambda_count"
                            ]
                        ),

                        lambda_game=float(
                            lcfg[
                                "lambda_game"
                            ]
                        ),

                        game_levels=tuple(
                            lcfg[
                                "game_levels"
                            ]
                        ),

                        game_level_weights=tuple(
                            lcfg[
                                "game_level_weights"
                            ]
                        ),
                    )
                )

                group_start = (
                    batch_idx
                    // accumulation_steps
                ) * accumulation_steps

                group_end = min(
                    group_start
                    + accumulation_steps,
                    len(train_loader),
                )

                group_size = (
                    group_end
                    - group_start
                )

                loss_for_backward = (
                    loss
                    / group_size
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            should_step = (
                (
                    batch_idx + 1
                )
                % accumulation_steps
                == 0
                or (
                    batch_idx + 1
                    == len(train_loader)
                )
            )

            if should_step:
                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            totals["loss"] += (
                logs["loss"]
            )

            totals["density"] += (
                logs["loss_density"]
            )

            totals["count"] += (
                logs["loss_count"]
            )

            totals["game"] += (
                logs["loss_game"]
            )

            n_batches += 1

            pbar.set_postfix(
                loss=(
                    totals["loss"]
                    / n_batches
                )
            )

        lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        train_loss = (
            totals["loss"]
            / n_batches
        )

        train_density = (
            totals["density"]
            / n_batches
        )

        train_count = (
            totals["count"]
            / n_batches
        )

        train_game = (
            totals["game"]
            / n_batches
        )

        val_mae = float("nan")
        val_game2 = float("nan")

        should_eval = (
            epoch % eval_every == 0
            or epoch == epochs
        )

        if should_eval:
            metrics = validate(
                model,
                val_loader,
                device,
            )

            val_mae = metrics["mae"]
            val_game2 = metrics["game2"]

            if val_game2 < best_game2:
                best_game2 = val_game2
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),

                        "optimizer_state_dict":
                            optimizer.state_dict(),

                        "scheduler_state_dict":
                            scheduler.state_dict(),

                        "scaler_state_dict":
                            scaler.state_dict(),

                        "val_game2":
                            val_game2,

                        "config":
                            cfg,
                    },
                    checkpoint_path,
                )

        with history_path.open(
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                lr,
                train_loss,
                train_density,
                train_count,
                train_game,
                val_mae,
                val_game2,
                best_game2,
            ])

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.6f} "
            f"GAME={train_game:.4f} "
            f"val_MAE={val_mae:.4f} "
            f"val_GAME2={val_game2:.4f} "
            f"best_GAME2={best_game2:.4f}"
        )

    summary = {
        "best_game2":
            best_game2,

        "best_game2_epoch":
            best_epoch,

        "checkpoint":
            str(checkpoint_path),
    }

    (
        run_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print(
        "\nTraining complete."
    )
    print(
        f"Best GAME2: {best_game2:.6f}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print(
        f"Checkpoint: {checkpoint_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment JSON config.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device, e.g. cuda, cuda:0, or cpu. "
            "Default: CUDA if available."
        ),
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

    train(
        cfg,
        device,
    )


if __name__ == "__main__":
    main()

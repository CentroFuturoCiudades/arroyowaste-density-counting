import torch
import torch.nn.functional as F


def _validate_density_pair(pred, target):
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "pred and target must have shape B x 1 x H x W"
        )

    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have identical shapes. "
            f"Got pred={tuple(pred.shape)} and "
            f"target={tuple(target.shape)}"
        )

    if pred.shape[1] != 1:
        raise ValueError(
            "density maps must have one channel"
        )


def game_metric(pred, target, level=1):
    """
    Canonical GAME-L.

    The image is divided into 4**level regions.
    Absolute count error is summed over all regions
    and averaged over the batch.
    """
    _validate_density_pair(pred, target)

    if not isinstance(level, int) or level < 0:
        raise ValueError(
            "level must be a non-negative integer"
        )

    batch_size, _, height, width = pred.shape
    splits = 2 ** level

    error = pred.new_zeros(batch_size)

    for row in range(splits):
        for column in range(splits):
            y0 = (row * height) // splits
            y1 = ((row + 1) * height) // splits
            x0 = (column * width) // splits
            x1 = ((column + 1) * width) // splits

            pred_count = pred[
                :, :, y0:y1, x0:x1
            ].sum(dim=(1, 2, 3))

            true_count = target[
                :, :, y0:y1, x0:x1
            ].sum(dim=(1, 2, 3))

            error += torch.abs(
                pred_count - true_count
            )

    return error.mean()


def regional_count_loss(
    pred,
    target,
    levels=(1, 2),
    weights=(1.0, 1.0),
):
    """
    GAME training term used in the paper.

    L_GAME =
        w1 * GAME-1
        + w2 * GAME-2

    No region or weight normalization is applied.
    """
    if len(levels) != len(weights):
        raise ValueError(
            "levels and weights must have equal length"
        )

    loss = pred.new_tensor(0.0)

    for level, weight in zip(levels, weights):
        loss = loss + float(weight) * game_metric(
            pred,
            target,
            level=int(level),
        )

    return loss


def density_count_loss(
    pred,
    target,
    density_scale=100.0,
    lambda_count=0.01,
    lambda_game=0.5,
    game_levels=(1, 2),
    game_level_weights=(1.0, 1.0),
):
    """
    Training objective used for the reported experiments.

    Density:
        MSE(100 * prediction, 100 * target)

    Count:
        L1 between total predicted and target counts.

    Regional:
        GAME-1 + GAME-2.

    Total:
        L_density
        + lambda_count * L_count
        + lambda_game * L_GAME
    """
    _validate_density_pair(pred, target)

    loss_density = F.mse_loss(
        pred * float(density_scale),
        target * float(density_scale),
    )

    pred_count = pred.sum(dim=(1, 2, 3))
    true_count = target.sum(dim=(1, 2, 3))

    loss_count = F.l1_loss(
        pred_count,
        true_count,
    )

    loss_game = regional_count_loss(
        pred,
        target,
        levels=game_levels,
        weights=game_level_weights,
    )

    weighted_count = (
        float(lambda_count)
        * loss_count
    )

    weighted_game = (
        float(lambda_game)
        * loss_game
    )

    loss = (
        loss_density
        + weighted_count
        + weighted_game
    )

    logs = {
        "loss": float(
            loss.detach().cpu()
        ),
        "loss_density": float(
            loss_density.detach().cpu()
        ),
        "loss_count": float(
            loss_count.detach().cpu()
        ),
        "loss_game": float(
            loss_game.detach().cpu()
        ),
        "weighted_count": float(
            weighted_count.detach().cpu()
        ),
        "weighted_game": float(
            weighted_game.detach().cpu()
        ),
    }

    return loss, logs

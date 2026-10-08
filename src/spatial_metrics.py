import math

import torch
import torch.nn.functional as F


def normalize_mass(
    x,
    eps=1e-12,
):
    x = torch.clamp(
        x.float(),
        min=0.0,
    )

    flat = x.reshape(
        x.shape[0],
        -1,
    )

    total = flat.sum(
        dim=1,
        keepdim=True,
    )

    return flat / (
        total + eps
    )


def mass_aware_downsample(
    x,
    out_hw=(48, 64),
):
    """
    Downsample a density map by regional averaging.

    The map is normalized afterwards for the spatial
    distribution metrics, so preserving relative regional
    mass is more important than preserving raw amplitudes.
    """

    return F.adaptive_avg_pool2d(
        x,
        output_size=out_hw,
    )


def make_cost_matrix(
    h,
    w,
    device,
    dtype=torch.float32,
):
    """
    Euclidean distance between normalized spatial coordinates.

    Coordinates lie in [0,1] x [0,1].
    Maximum distance is sqrt(2).
    """

    ys = torch.linspace(
        0.0,
        1.0,
        h,
        device=device,
        dtype=dtype,
    )

    xs = torch.linspace(
        0.0,
        1.0,
        w,
        device=device,
        dtype=dtype,
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    coords = torch.stack(
        [
            yy.reshape(-1),
            xx.reshape(-1),
        ],
        dim=1,
    )

    return torch.cdist(
        coords,
        coords,
        p=2,
    )


def regularized_ot_cost(
    a,
    b,
    C,
    epsilon=0.05,
    n_iters=200,
    eps=1e-12,
):
    """
    Entropic balanced OT objective for probability vectors.

    Returns one value per image.
    """

    K = torch.exp(
        -C / epsilon
    ).clamp_min(
        eps
    )

    B, N = a.shape

    u = torch.ones_like(a) / N
    v = torch.ones_like(b) / N

    for _ in range(
        n_iters
    ):

        Kv = torch.matmul(
            K,
            v.unsqueeze(-1),
        ).squeeze(-1)

        u = a / (
            Kv + eps
        )

        KTu = torch.matmul(
            K.t(),
            u.unsqueeze(-1),
        ).squeeze(-1)

        v = b / (
            KTu + eps
        )

    P = (
        u.unsqueeze(2)
        * K.unsqueeze(0)
        * v.unsqueeze(1)
    )

    transport_term = (
        P
        * C.unsqueeze(0)
    ).sum(
        dim=(1, 2)
    )

    entropy_term = epsilon * (
        P
        * (
            torch.log(
                P.clamp_min(eps)
            )
            - 1.0
        )
    ).sum(
        dim=(1, 2)
    )

    return (
        transport_term
        + entropy_term
    )


@torch.no_grad()
def spatial_sinkhorn_divergence(
    pred,
    target,
    out_hw=(48, 64),
    epsilon=0.05,
    n_iters=200,
):
    pred_small = (
        mass_aware_downsample(
            pred,
            out_hw=out_hw,
        )
    )

    target_small = (
        mass_aware_downsample(
            target,
            out_hw=out_hw,
        )
    )

    a = normalize_mass(
        pred_small
    )

    b = normalize_mass(
        target_small
    )

    h, w = out_hw

    C = make_cost_matrix(
        h,
        w,
        pred.device,
    )

    ab = regularized_ot_cost(
        a,
        b,
        C,
        epsilon=epsilon,
        n_iters=n_iters,
    )

    aa = regularized_ot_cost(
        a,
        a,
        C,
        epsilon=epsilon,
        n_iters=n_iters,
    )

    bb = regularized_ot_cost(
        b,
        b,
        C,
        epsilon=epsilon,
        n_iters=n_iters,
    )

    value = (
        ab
        - 0.5 * aa
        - 0.5 * bb
    )

    return torch.clamp(
        value,
        min=0.0,
    )

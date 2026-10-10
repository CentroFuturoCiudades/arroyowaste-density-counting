import numpy as np
import cv2


def make_density_map(
    points,
    height,
    width,
    sigma=4.0,
    out_stride=4,
    adaptive=False,
    beta=0.3,
    k_neighbors=3,
    sigma_min=1.0,
    sigma_max=12.0,
):
    """
    Create a density map from point annotations.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), with points as (x, y) in image coordinates.
    height, width : int
        Image size after resizing.
    sigma : float
        Fixed Gaussian sigma. Used if adaptive=False.
    out_stride : int
        Output stride of the model. For example, 4 means H/4 × W/4.
    adaptive : bool
        If True, use a simplified geometry-adaptive sigma inspired by MCNN:
        sigma_i = beta * mean distance to k nearest neighbors.
        If False, use fixed sigma.
    beta : float
        Scale factor for adaptive sigma.
    k_neighbors : int
        Number of neighbors used for adaptive sigma.
    sigma_min : float
        Minimum allowed adaptive sigma.
    sigma_max : float
        Maximum allowed adaptive sigma.

    Returns
    -------
    density_small : np.ndarray
        Density map of shape (height/out_stride, width/out_stride).
        Its sum is exactly approximately equal to number of points.
    """

    points = np.asarray(points, dtype=np.float32)

    if height <= 0 or width <= 0:
        raise ValueError(
            f"height and width must be positive, got "
            f"{(height, width)}"
        )

    if out_stride <= 0:
        raise ValueError(
            f"out_stride must be positive, got {out_stride}"
        )

    if height % out_stride != 0 or width % out_stride != 0:
        raise ValueError(
            f"Image shape {(height, width)} must be divisible "
            f"by out_stride={out_stride}"
        )

    if sigma <= 0:
        raise ValueError(
            f"sigma must be positive, got {sigma}"
        )

    if beta <= 0:
        raise ValueError(
            f"beta must be positive, got {beta}"
        )

    if k_neighbors < 1:
        raise ValueError(
            f"k_neighbors must be >= 1, got {k_neighbors}"
        )

    if sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError(
            f"Invalid sigma limits: "
            f"sigma_min={sigma_min}, sigma_max={sigma_max}"
        )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {points.shape}")

    target_count = float(len(points))

    density = np.zeros((height, width), dtype=np.float32)

    if target_count == 0:
        return np.zeros(
            (height // out_stride, width // out_stride),
            dtype=np.float32,
        )

    # Keep only points inside the resized image.
    valid_points = []
    for x, y in points:
        xi = int(round(float(x)))
        yi = int(round(float(y)))

        if 0 <= xi < width and 0 <= yi < height:
            valid_points.append((float(x), float(y), xi, yi))

    if len(valid_points) == 0:
        return np.zeros(
            (height // out_stride, width // out_stride),
            dtype=np.float32,
        )

    # Fixed Gaussian version: recommended first for Part B.
    if not adaptive:
        for _, _, xi, yi in valid_points:
            density[yi, xi] += 1.0

        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        density = cv2.GaussianBlur(
            density,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_CONSTANT,
        )

    # Simplified geometry-adaptive version inspired by MCNN.
    else:
        coords = np.array([[x, y] for x, y, _, _ in valid_points], dtype=np.float32)
        n = len(coords)

        for idx, (x, y, xi, yi) in enumerate(valid_points):
            if n > 1:
                dists = np.sqrt(np.sum((coords - coords[idx]) ** 2, axis=1))
                dists = np.sort(dists)[1 : min(k_neighbors + 1, n)]
                if len(dists) > 0:
                    sigma_i = beta * float(np.mean(dists))
                else:
                    sigma_i = sigma
            else:
                sigma_i = sigma

            sigma_i = max(
                float(sigma_min),
                min(float(sigma_i), float(sigma_max)),
            )

            radius = int(3 * sigma_i)
            kernel_size = 2 * radius + 1

            x0 = max(0, xi - radius)
            x1 = min(width, xi + radius + 1)
            y0 = max(0, yi - radius)
            y1 = min(height, yi + radius + 1)

            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_i ** 2))
            g_sum = g.sum()

            if g_sum > 1e-8:
                g /= g_sum

            density[y0:y1, x0:x1] += g.astype(np.float32)

    # Downsample to model output resolution.
    h2 = height // out_stride
    w2 = width // out_stride

    density_small = cv2.resize(
        density,
        (w2, h2),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)

    # Critical: preserve total count after blur + borders + downsampling.
    current_count = float(density_small.sum())

    if current_count > 1e-8:
        density_small *= target_count / current_count

    return density_small.astype(np.float32)

#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


EXPECTED_SAMPLES = 640
EXPECTED_POINTS = 3260
MIN_COMPONENT_AREA = 20


def prepare_output(output):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"{output} already exists and is not empty."
        )

    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(parents=True, exist_ok=True)


def load_selection(manifest):
    data = pd.read_csv(manifest, dtype=str)
    selected = data.loc[
        data["dataset_source"] == "dswd",
        "original_id",
    ].tolist()

    if len(selected) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} DSWD samples, "
            f"found {len(selected)} in {manifest}."
        )

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate DSWD IDs in manifest.")

    return selected


def find_image(directory, stem):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = directory / f"{stem}{suffix}"

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Image not found for {stem} in {directory}"
    )


def mask_to_centers(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    binary = (mask > 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    centers = []

    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            x, y = centroids[index]
            centers.append((float(x), float(y)))

    return centers


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen DSWD training subset used by ABD."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/abd.csv"),
    )
    args = parser.parse_args()

    selected = load_selection(args.manifest)
    prepare_output(args.output)
    total_points = 0

    for sample_id in selected:
        if not sample_id.startswith("img_"):
            raise ValueError(f"Unexpected DSWD ID: {sample_id}")

        source_image = find_image(args.images, sample_id)
        mask_id = "mask_" + sample_id.removeprefix("img_")
        source_mask = find_image(args.masks, mask_id)
        centers = mask_to_centers(source_mask)

        shutil.copy2(
            source_image,
            args.output / "images" / source_image.name,
        )
        np.save(
            args.output / "annotations" / f"{sample_id}.npy",
            centers,
        )
        total_points += len(centers)

    if total_points != EXPECTED_POINTS:
        raise ValueError(
            f"Expected {EXPECTED_POINTS} DSWD points, "
            f"generated {total_points}."
        )

    print(f"DSWD samples: {len(selected)}")
    print(f"DSWD points:  {total_points}")


if __name__ == "__main__":
    main()

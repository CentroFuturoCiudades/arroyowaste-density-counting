#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify an ArroyoWaste or ABD "
            "dataset against its manifest."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.manifest
    )

    print(
        f"Manifest: {args.manifest}"
    )
    print(
        f"Dataset:  {args.root}"
    )
    print(
        f"Samples:  {len(df)}"
    )

    missing = []
    bad_counts = []
    bad_image_hash = []
    bad_points_hash = []

    for _, row in df.iterrows():
        image = (
            args.root
            / str(row["image_file"])
        )

        points = (
            args.root
            / str(row["points_file"])
        )

        if not image.is_file():
            missing.append(
                str(image)
            )
            continue

        if not points.is_file():
            missing.append(
                str(points)
            )
            continue

        annotations = np.load(
            points,
            allow_pickle=False,
        )

        if len(annotations) != int(
            row["count"]
        ):
            bad_counts.append(
                str(row["sample_id"])
            )

        if (
            sha256_file(image)
            != str(row["image_sha256"])
        ):
            bad_image_hash.append(
                str(row["sample_id"])
            )

        if (
            sha256_file(points)
            != str(row["points_sha256"])
        ):
            bad_points_hash.append(
                str(row["sample_id"])
            )

    print()
    print("Splits:")
    print(
        df["split"]
        .value_counts()
        .sort_index()
    )

    if "dataset_source" in df.columns:
        print()
        print("Sources:")
        print(
            df["dataset_source"]
            .value_counts()
            .sort_index()
        )

    print()
    print(
        "Total annotations:",
        int(df["count"].sum()),
    )

    if missing:
        raise RuntimeError(
            f"Missing files: {len(missing)}\n"
            f"First: {missing[0]}"
        )

    if bad_counts:
        raise RuntimeError(
            "Annotation count mismatch: "
            f"{bad_counts[0]}"
        )

    if bad_image_hash:
        raise RuntimeError(
            "Image SHA256 mismatch: "
            f"{bad_image_hash[0]}"
        )

    if bad_points_hash:
        raise RuntimeError(
            "Points SHA256 mismatch: "
            f"{bad_points_hash[0]}"
        )

    print()
    print("Files:          PASS")
    print("Point counts:   PASS")
    print("Image SHA256:   PASS")
    print("Points SHA256:  PASS")
    print()
    print("Dataset verification: PASS")


if __name__ == "__main__":
    main()

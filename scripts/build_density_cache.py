#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from data.density_map import make_density_map


RECIPE_ID = (
    "adaptive_b03_k3_s40_"
    "clip10-120_stride4_native"
)

EXPECTED = {
    "arroyowaste": {
        "samples": 414,
        "splits": {
            "train": 392,
            "val": 11,
            "test": 11,
        },
    },
    "abd": {
        "samples": 1452,
        "splits": {
            "train": 1430,
            "val": 11,
            "test": 11,
        },
    },
}


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            "Cannot write empty CSV."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def atomic_save_npy(
    path,
    array,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("wb") as stream:
        np.save(
            stream,
            array,
            allow_pickle=False,
        )

    loaded = np.load(
        temporary,
        allow_pickle=False,
    )

    np.testing.assert_array_equal(
        loaded,
        array,
    )

    os.replace(
        temporary,
        path,
    )


def validate_points(
    points,
    width,
    height,
    sample_id,
):
    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        raise ValueError(
            f"{sample_id}: invalid "
            f"point shape {points.shape}"
        )

    if not np.isfinite(
        points
    ).all():
        raise ValueError(
            f"{sample_id}: non-finite "
            "point coordinates"
        )

    if len(points) == 0:
        return

    invalid = (
        (points[:, 0] < 0)
        | (points[:, 0] >= width)
        | (points[:, 1] < 0)
        | (points[:, 1] >= height)
    )

    if invalid.any():
        raise ValueError(
            f"{sample_id}: "
            f"{int(invalid.sum())} "
            "points outside image bounds"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the density-map cache "
            "used by the ArroyoWaste "
            "density-counting experiments."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "arroyowaste",
            "abd",
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dataset_root is None:
        dataset_root = (
            REPO_ROOT
            / "data"
            / args.dataset
        )
    else:
        dataset_root = (
            args.dataset_root
            .expanduser()
            .resolve()
        )

    if args.manifest is None:
        manifest_path = (
            REPO_ROOT
            / "manifests"
            / f"{args.dataset}.csv"
        )
    else:
        manifest_path = (
            args.manifest
            .expanduser()
            .resolve()
        )

    dataset_root = (
        dataset_root.resolve()
    )

    manifest_path = (
        manifest_path.resolve()
    )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            dataset_root
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    rows = load_csv(
        manifest_path
    )

    profile = EXPECTED[
        args.dataset
    ]

    if (
        len(rows)
        != profile["samples"]
    ):
        raise ValueError(
            f"{args.dataset}: expected "
            f"{profile['samples']} samples, "
            f"found {len(rows)}"
        )

    split_counts = Counter(
        row["split"]
        for row in rows
    )

    if (
        dict(split_counts)
        != profile["splits"]
    ):
        raise ValueError(
            "Unexpected split counts: "
            f"{dict(split_counts)}"
        )

    required_columns = {
        "sample_id",
        "source_id",
        "image_file",
        "points_file",
        "width",
        "height",
        "count",
        "split",
    }

    missing = (
        required_columns
        - set(rows[0])
    )

    if missing:
        raise ValueError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    output_root = (
        dataset_root
        / "density_maps"
        / RECIPE_ID
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_rows = []

    created = 0
    skipped = 0
    total_objects = 0
    maximum_mass_error = 0.0

    seen = set()

    for index, row in enumerate(
        rows,
        start=1,
    ):
        sample_id = row[
            "sample_id"
        ]

        if sample_id in seen:
            raise ValueError(
                f"Duplicate sample_id: "
                f"{sample_id}"
            )

        seen.add(sample_id)

        split = row["split"]

        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"{sample_id}: "
                f"invalid split {split!r}"
            )

        image_path = (
            dataset_root
            / row["image_file"]
        )

        points_path = (
            dataset_root
            / row["points_file"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                image_path
            )

        if not points_path.is_file():
            raise FileNotFoundError(
                points_path
            )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Unreadable image: "
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        if (
            width != int(row["width"])
            or height
            != int(row["height"])
        ):
            raise ValueError(
                f"{sample_id}: image "
                "dimensions differ from "
                "manifest"
            )

        points = np.asarray(
            np.load(
                points_path,
                allow_pickle=False,
            ),
            dtype=np.float32,
        ).reshape(-1, 2)

        expected_count = int(
            row["count"]
        )

        if (
            len(points)
            != expected_count
        ):
            raise ValueError(
                f"{sample_id}: "
                f"{len(points)} points, "
                f"expected {expected_count}"
            )

        validate_points(
            points,
            width,
            height,
            sample_id,
        )

        density_path = (
            output_root
            / split
            / f"{sample_id}.npy"
        )

        if (
            density_path.exists()
            and not args.overwrite
        ):
            density = np.asarray(
                np.load(
                    density_path,
                    allow_pickle=False,
                ),
                dtype=np.float32,
            )

            skipped += 1

        else:
            density = make_density_map(
                points=points,
                height=height,
                width=width,
                sigma=4.0,
                out_stride=4,
                adaptive=True,
                beta=0.3,
                k_neighbors=3,
                sigma_min=1.0,
                sigma_max=12.0,
            )

            density = np.asarray(
                density,
                dtype=np.float32,
            )

            atomic_save_npy(
                density_path,
                density,
            )

            created += 1

        expected_shape = (
            height // 4,
            width // 4,
        )

        if (
            density.shape
            != expected_shape
        ):
            raise ValueError(
                f"{sample_id}: density "
                f"shape {density.shape}, "
                f"expected "
                f"{expected_shape}"
            )

        if not np.isfinite(
            density
        ).all():
            raise ValueError(
                f"{sample_id}: non-finite "
                "density values"
            )

        if np.any(density < 0):
            raise ValueError(
                f"{sample_id}: negative "
                "density values"
            )

        density_sum = float(
            density.sum(
                dtype=np.float64
            )
        )

        mass_error = abs(
            density_sum
            - expected_count
        )

        tolerance = max(
            1e-4,
            expected_count * 1e-6,
        )

        if mass_error > tolerance:
            raise ValueError(
                f"{sample_id}: density "
                f"mass error "
                f"{mass_error:.6e}"
            )

        maximum_mass_error = max(
            maximum_mass_error,
            mass_error,
        )

        total_objects += (
            expected_count
        )

        density_relative = (
            density_path
            .relative_to(
                dataset_root
            )
            .as_posix()
        )

        cache_rows.append({
            "sample_id":
                sample_id,

            "source_id":
                row["source_id"],

            "experiment_split":
                split,

            "image_path":
                row["image_file"],

            "points_path":
                row["points_file"],

            "density_path":
                density_relative,

            "original_height":
                height,

            "original_width":
                width,

            "processed_height":
                height,

            "processed_width":
                width,

            "density_height":
                density.shape[0],

            "density_width":
                density.shape[1],

            "count":
                expected_count,

            "density_sum":
                f"{density_sum:.12f}",

            "mass_error":
                f"{mass_error:.12e}",

            "image_sha256":
                sha256_file(
                    image_path
                ),

            "points_sha256":
                sha256_file(
                    points_path
                ),

            "density_sha256":
                sha256_file(
                    density_path
                ),

            "recipe_id":
                RECIPE_ID,
        })

        if (
            index % 100 == 0
            or index == len(rows)
        ):
            print(
                f"{index:4d}/"
                f"{len(rows)} "
                "density maps verified"
            )

    cache_manifest = (
        output_root
        / "density_manifest.csv"
    )

    write_csv(
        cache_manifest,
        cache_rows,
    )

    metadata = {
        "dataset":
            (
                "ArroyoWaste"
                if args.dataset
                == "arroyowaste"
                else "ABD"
            ),

        "recipe_id":
            RECIPE_ID,

        "samples":
            len(rows),

        "split_counts":
            dict(split_counts),

        "total_objects":
            total_objects,

        "benchmark_manifest":
            str(
                manifest_path
                .relative_to(
                    REPO_ROOT
                )
            )
            if manifest_path
            .is_relative_to(
                REPO_ROOT
            )
            else str(
                manifest_path
            ),

        "benchmark_manifest_sha256":
            sha256_file(
                manifest_path
            ),

        "parameters": {
            "resize_mode":
                "none",
            "out_stride":
                4,
            "sigma":
                4.0,
            "adaptive":
                True,
            "beta":
                0.3,
            "k_neighbors":
                3,
            "sigma_min":
                1.0,
            "sigma_max":
                12.0,
        },

        "density_shape":
            [192, 256],

        "maximum_mass_error":
            maximum_mass_error,
    }

    metadata_path = (
        output_root
        / "metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Density cache")
    print("-------------")
    print(
        "dataset:      ",
        metadata["dataset"],
    )
    print(
        "recipe:       ",
        RECIPE_ID,
    )
    print(
        "samples:      ",
        len(rows),
    )
    print(
        "created:      ",
        created,
    )
    print(
        "skipped:      ",
        skipped,
    )
    print(
        "objects:      ",
        total_objects,
    )
    print(
        "max mass err: ",
        maximum_mass_error,
    )
    print()
    print(
        "Density cache build: PASS"
    )


if __name__ == "__main__":
    main()

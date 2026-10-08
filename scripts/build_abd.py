#!/usr/bin/env python3

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


TARGET_WIDTH = 1024
TARGET_HEIGHT = 768

EXPECTED_TOTAL = 1452
EXPECTED_TRAIN = 1430
EXPECTED_VAL = 11
EXPECTED_TEST = 11

EXPECTED_ARROYOWASTE = 414
EXPECTED_BEPLI = 398
EXPECTED_DSWD = 640


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_points(path):
    points = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        ),
        dtype=np.float32,
    )

    if points.size == 0:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        raise ValueError(
            f"Invalid point shape "
            f"{points.shape}: {path}"
        )

    if not np.isfinite(points).all():
        raise ValueError(
            f"Non-finite points: {path}"
        )

    return points


def validate_points(
    points,
    width,
    height,
    label,
):
    if len(points) == 0:
        return

    x = points[:, 0]
    y = points[:, 1]

    valid = (
        (x >= 0).all()
        and (x < width).all()
        and (y >= 0).all()
        and (y < height).all()
    )

    if not valid:
        raise ValueError(
            f"Points outside image bounds: "
            f"{label}"
        )


def find_image(
    directory,
    stem,
):
    for suffix in (
        ".png",
        ".jpg",
        ".jpeg",
    ):
        path = directory / f"{stem}{suffix}"

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Image not found for {stem} "
        f"in {directory}"
    )


def prepare_output(output):
    if output.exists():
        contents = list(
            output.iterdir()
        )

        if contents:
            raise RuntimeError(
                f"{output} already exists "
                "and is not empty."
            )

    (
        output
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output
        / "annotations"
        / "points"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def build_arroyowaste_index(
    manifest_path,
):
    df = pd.read_csv(
        manifest_path
    )

    if "sample_id" not in df.columns:
        raise ValueError(
            "ArroyoWaste manifest must "
            "contain sample_id"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample_id in "
            "ArroyoWaste manifest"
        )

    return {
        str(row["sample_id"]): row
        for _, row in df.iterrows()
    }


def build_arroyowaste_sample(
    row,
    arroyowaste_root,
    arroyowaste_index,
    output_root,
):
    original_id = str(
        row["original_id"]
    )

    if original_id not in arroyowaste_index:
        raise KeyError(
            f"ArroyoWaste sample not found: "
            f"{original_id}"
        )

    source = (
        arroyowaste_index[
            original_id
        ]
    )

    source_image = (
        arroyowaste_root
        / str(source["image_file"])
    )

    source_points = (
        arroyowaste_root
        / str(source["points_file"])
    )

    if not source_image.is_file():
        raise FileNotFoundError(
            source_image
        )

    if not source_points.is_file():
        raise FileNotFoundError(
            source_points
        )

    image = cv2.imread(
        str(source_image),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unreadable image: "
            f"{source_image}"
        )

    height, width = image.shape[:2]

    if (
        width != TARGET_WIDTH
        or height != TARGET_HEIGHT
    ):
        raise ValueError(
            f"Unexpected ArroyoWaste size "
            f"{width}x{height}: "
            f"{source_image}"
        )

    points = load_points(
        source_points
    )

    validate_points(
        points,
        width,
        height,
        str(source_image),
    )

    output_image = (
        output_root
        / str(row["image_file"])
    )

    output_points = (
        output_root
        / str(row["points_file"])
    )

    output_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_points.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_image,
        output_image,
    )

    np.save(
        output_points,
        points.astype(
            np.float32,
            copy=False,
        ),
    )


def build_bepli_sample(
    row,
    bepli_root,
    output_root,
):
    original_id = str(
        row["original_id"]
    )

    image_dir = (
        bepli_root
        / "train_data"
        / "images"
    )

    points_dir = (
        bepli_root
        / "train_data"
        / "annotations"
    )

    source_image = find_image(
        image_dir,
        original_id,
    )

    source_points = (
        points_dir
        / f"{original_id}.npy"
    )

    if not source_points.is_file():
        raise FileNotFoundError(
            source_points
        )

    image = cv2.imread(
        str(source_image),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unreadable BePLi image: "
            f"{source_image}"
        )

    height, width = image.shape[:2]

    if (
        width != TARGET_WIDTH
        or height != TARGET_HEIGHT
    ):
        raise ValueError(
            "BePLi image is not "
            f"1024x768: {source_image}"
        )

    points = load_points(
        source_points
    )

    validate_points(
        points,
        width,
        height,
        str(source_image),
    )

    output_image = (
        output_root
        / str(row["image_file"])
    )

    output_points = (
        output_root
        / str(row["points_file"])
    )

    output_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_points.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_image,
        output_image,
    )

    np.save(
        output_points,
        points.astype(
            np.float32,
            copy=False,
        ),
    )


def build_dswd_sample(
    row,
    dswd_root,
    output_root,
):
    original_id = str(
        row["original_id"]
    )

    image_dir = (
        dswd_root
        / "train_data"
        / "images"
    )

    points_dir = (
        dswd_root
        / "train_data"
        / "annotations"
    )

    source_image = find_image(
        image_dir,
        original_id,
    )

    source_points = (
        points_dir
        / f"{original_id}.npy"
    )

    if not source_points.is_file():
        raise FileNotFoundError(
            source_points
        )

    image = cv2.imread(
        str(source_image),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unreadable DSWD image: "
            f"{source_image}"
        )

    height, width = image.shape[:2]

    if (
        width != 448
        or height != 448
    ):
        raise ValueError(
            "Expected DSWD source "
            f"448x448, got "
            f"{width}x{height}: "
            f"{source_image}"
        )

    points = load_points(
        source_points
    )

    validate_points(
        points,
        width,
        height,
        str(source_image),
    )

    scale = (
        TARGET_HEIGHT
        / 448.0
    )

    resized_width = 768
    resized_height = 768
    pad_left = 128

    resized = cv2.resize(
        image,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros(
        (
            TARGET_HEIGHT,
            TARGET_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :,
        pad_left:
        pad_left + resized_width,
    ] = resized

    transformed_points = (
        points.copy()
    )

    if len(transformed_points):
        transformed_points[:, 0] = (
            transformed_points[:, 0]
            * scale
            + pad_left
        )

        transformed_points[:, 1] = (
            transformed_points[:, 1]
            * scale
        )

    validate_points(
        transformed_points,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        original_id,
    )

    output_image = (
        output_root
        / str(row["image_file"])
    )

    output_points = (
        output_root
        / str(row["points_file"])
    )

    output_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_points.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_image),
        canvas,
    )

    if not success:
        raise RuntimeError(
            f"Could not write "
            f"{output_image}"
        )

    np.save(
        output_points,
        transformed_points.astype(
            np.float32,
            copy=False,
        ),
    )


def verify_sample(
    row,
    output_root,
):
    image_path = (
        output_root
        / str(row["image_file"])
    )

    points_path = (
        output_root
        / str(row["points_file"])
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
            f"Unreadable output image: "
            f"{image_path}"
        )

    height, width = image.shape[:2]

    if (
        width != int(row["width"])
        or height != int(row["height"])
    ):
        raise ValueError(
            f"Image size mismatch for "
            f"{row['sample_id']}"
        )

    points = load_points(
        points_path
    )

    if len(points) != int(
        row["count"]
    ):
        raise ValueError(
            f"Point count mismatch for "
            f"{row['sample_id']}: "
            f"{len(points)} != "
            f"{row['count']}"
        )

    expected_image_hash = str(
        row["image_sha256"]
    )

    actual_image_hash = sha256_file(
        image_path
    )

    if (
        actual_image_hash
        != expected_image_hash
    ):
        raise ValueError(
            "Image SHA256 mismatch for "
            f"{row['sample_id']}\n"
            f"expected: "
            f"{expected_image_hash}\n"
            f"actual:   "
            f"{actual_image_hash}"
        )

    expected_points_hash = str(
        row["points_sha256"]
    )

    actual_points_hash = sha256_file(
        points_path
    )

    if (
        actual_points_hash
        != expected_points_hash
    ):
        raise ValueError(
            "Points SHA256 mismatch for "
            f"{row['sample_id']}\n"
            f"expected: "
            f"{expected_points_hash}\n"
            f"actual:   "
            f"{actual_points_hash}"
        )


def verify_contract(df):
    if len(df) != EXPECTED_TOTAL:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL} "
            f"samples, got {len(df)}"
        )

    split_counts = (
        df["split"]
        .value_counts()
        .to_dict()
    )

    expected_splits = {
        "train": EXPECTED_TRAIN,
        "val": EXPECTED_VAL,
        "test": EXPECTED_TEST,
    }

    if split_counts != expected_splits:
        raise ValueError(
            "Unexpected split counts: "
            f"{split_counts}"
        )

    sources = (
        df["dataset_source"]
        .replace({
            "arroyowaste":
                "arroyowaste",
        })
        .value_counts()
        .to_dict()
    )

    expected_sources = {
        "arroyowaste":
            EXPECTED_ARROYOWASTE,

        "bepli":
            EXPECTED_BEPLI,

        "dswd":
            EXPECTED_DSWD,
    }

    if sources != expected_sources:
        raise ValueError(
            "Unexpected source counts: "
            f"{sources}"
        )

    if df["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample_id in "
            "ABD manifest"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the exact ABD "
            "corpus used in the paper."
        )
    )

    parser.add_argument(
        "--arroyowaste",
        type=Path,
        default=Path(
            "data/arroyowaste"
        ),
    )

    parser.add_argument(
        "--arroyowaste-manifest",
        type=Path,
        default=Path(
            "manifests/arroyowaste.csv"
        ),
    )

    parser.add_argument(
        "--bepli",
        type=Path,
        default=Path(
            "data/sources/bepli"
        ),
    )

    parser.add_argument(
        "--dswd",
        type=Path,
        default=Path(
            "data/sources/dswd"
        ),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "manifests/abd.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/abd"
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(
        args.manifest
    )

    verify_contract(df)

    arroyo_index = (
        build_arroyowaste_index(
            args.arroyowaste_manifest
        )
    )

    prepare_output(
        args.output
    )

    counts = {
        "arroyowaste": 0,
        "bepli": 0,
        "dswd": 0,
    }

    total = len(df)

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        source = str(
            row["dataset_source"]
        )

        if source == "arroyowaste":
            build_arroyowaste_sample(
                row,
                args.arroyowaste,
                arroyo_index,
                args.output,
            )

            counts[
                "arroyowaste"
            ] += 1

        elif source == "bepli":
            build_bepli_sample(
                row,
                args.bepli,
                args.output,
            )

            counts["bepli"] += 1

        elif source == "dswd":
            build_dswd_sample(
                row,
                args.dswd,
                args.output,
            )

            counts["dswd"] += 1

        else:
            raise ValueError(
                f"Unknown source: {source}"
            )

        verify_sample(
            row,
            args.output,
        )

        if (
            position % 100 == 0
            or position == total
        ):
            print(
                f"{position:4d}/{total} "
                "verified"
            )

    manifest_dir = (
        args.output
        / "manifests"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        args.manifest,
        manifest_dir / "abd.csv",
    )

    print()
    print("ABD reconstruction")
    print("------------------")
    print(
        "ArroyoWaste:",
        f"{counts['arroyowaste']}/"
        f"{EXPECTED_ARROYOWASTE}",
    )
    print(
        "BePLi:      ",
        f"{counts['bepli']}/"
        f"{EXPECTED_BEPLI}",
    )
    print(
        "DSWD:       ",
        f"{counts['dswd']}/"
        f"{EXPECTED_DSWD}",
    )

    print()
    print(
        "Train:      ",
        int(
            (df["split"] == "train")
            .sum()
        ),
    )
    print(
        "Validation: ",
        int(
            (df["split"] == "val")
            .sum()
        ),
    )
    print(
        "Test:       ",
        int(
            (df["split"] == "test")
            .sum()
        ),
    )
    print(
        "Total:      ",
        len(df),
    )

    print()
    print(
        "Image dimensions: PASS"
    )
    print(
        "Point counts:     PASS"
    )
    print(
        "Image SHA256:     PASS"
    )
    print(
        "Points SHA256:    PASS"
    )

    print()
    print(
        "ABD reconstruction: PASS"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils


TARGET_WIDTH = 1024
TARGET_HEIGHT = 768
EXPECTED_SAMPLES = 398
EXPECTED_POINTS = 10306


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
        data["dataset_source"] == "bepli",
        "original_id",
    ].tolist()

    if len(selected) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} BePLi samples, "
            f"found {len(selected)} in {manifest}."
        )

    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate BePLi IDs in manifest.")

    return selected


def load_coco(annotation_dir):
    images = {}
    annotations = []

    for split in ("train", "val", "test"):
        path = annotation_dir / f"{split}.json"

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)

        for image in data["images"]:
            image_id = int(image["id"])

            if image_id in images:
                raise ValueError(
                    f"Duplicate COCO image ID: {image_id}"
                )

            images[image_id] = image

        annotations.extend(data["annotations"])

    annotations.sort(key=lambda item: int(item["id"]))

    by_image = {}

    for annotation in annotations:
        image_id = int(annotation["image_id"])
        by_image.setdefault(image_id, []).append(annotation)

    by_filename = {
        str(image["file_name"]): image
        for image in images.values()
    }

    return by_filename, by_image


def transform_image(image):
    height, width = image.shape[:2]

    if height > width:
        image = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    height, width = image.shape[:2]
    scale = min(
        TARGET_WIDTH / float(width),
        TARGET_HEIGHT / float(height),
    )
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    pad_width = TARGET_WIDTH - new_width
    pad_height = TARGET_HEIGHT - new_height

    return cv2.copyMakeBorder(
        image,
        pad_height // 2,
        pad_height - pad_height // 2,
        pad_width // 2,
        pad_width - pad_width // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def transform_mask(mask, rotate):
    if rotate:
        mask = np.rot90(mask, k=3)

    height, width = mask.shape[:2]
    scale = min(
        TARGET_WIDTH / float(width),
        TARGET_HEIGHT / float(height),
    )

    if scale != 1.0:
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        mask = cv2.resize(
            mask.astype(np.uint8),
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )

    height, width = mask.shape[:2]
    pad_width = TARGET_WIDTH - width
    pad_height = TARGET_HEIGHT - height

    return np.pad(
        mask,
        (
            (
                pad_height // 2,
                pad_height - pad_height // 2,
            ),
            (
                pad_width // 2,
                pad_width - pad_width // 2,
            ),
        ),
        mode="constant",
        constant_values=0,
    )


def mask_centroid(annotation, rotate):
    segmentation = dict(annotation["segmentation"])

    if isinstance(segmentation["counts"], str):
        segmentation["counts"] = segmentation[
            "counts"
        ].encode("utf-8")

    mask = mask_utils.decode(segmentation).astype(np.uint8)
    mask = transform_mask(mask, rotate)
    y, x = np.where(mask > 0)

    if not y.size:
        return None

    return [float(x.mean()), float(y.mean())]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen BePLi source subset used by ABD."
        )
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/abd.csv"),
    )
    args = parser.parse_args()

    selected = load_selection(args.manifest)
    images, annotations = load_coco(args.annotations)
    prepare_output(args.output)

    total_points = 0

    for sample_id in selected:
        filename = f"{sample_id}.png"
        source = args.images / filename

        if filename not in images:
            raise KeyError(f"Missing COCO image entry: {filename}")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Cannot read image: {source}")

        info = images[filename]

        if image.shape[:2] != (
            int(info["height"]),
            int(info["width"]),
        ):
            raise ValueError(
                f"Image dimensions do not match COCO metadata: {source}"
            )

        rotate = int(info["height"]) > int(info["width"])
        output_image = transform_image(image)
        points = []

        for annotation in annotations.get(int(info["id"]), []):
            center = mask_centroid(annotation, rotate)

            if center is not None:
                points.append(center)

        points = np.asarray(points, dtype=np.float32)

        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float32)

        written = cv2.imwrite(
            str(args.output / "images" / filename),
            output_image,
        )

        if not written:
            raise RuntimeError(f"Cannot write image: {filename}")

        np.save(
            args.output / "annotations" / f"{sample_id}.npy",
            points,
        )
        total_points += len(points)

    if total_points != EXPECTED_POINTS:
        raise ValueError(
            f"Expected {EXPECTED_POINTS} BePLi points, "
            f"generated {total_points}."
        )

    print(f"BePLi samples: {len(selected)}")
    print(f"BePLi points:  {total_points}")


if __name__ == "__main__":
    main()

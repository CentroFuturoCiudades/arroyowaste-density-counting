from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ABDDataset(Dataset):
    """
    ABD benchmark dataset.

    Primary comparison contract
    ---------------------------
    train:
        ArroyoWaste train
        + BePLi train
        + DSWD train

    val:
        ArroyoWaste val only

    test:
        ArroyoWaste test only

    Canonical tensors
    -----------------
    image:
        float32 [3, 768, 1024], RGB, normalized to [0, 1]

    density:
        float32 [1, 192, 256]

    The dataset never regenerates density maps during normal training when
    density_cache_mode="require".
    """

    EXPECTED_SPLITS = {
        "train": 1430,
        "val": 11,
        "test": 11,
    }

    EXPECTED_IMAGE_HW = (768, 1024)
    EXPECTED_DENSITY_HW = (192, 256)

    EXPECTED_RECIPE = (
        "adaptive_b03_k3_s40_clip10-120_stride4_native"
    )

    def __init__(
        self,
        root,
        split="train",
        split_manifest="manifests/benchmark_seed7.csv",
        out_stride=4,
        sigma=4.0,
        adaptive=True,
        beta=0.3,
        k_neighbors=3,
        sigma_min=1.0,
        sigma_max=12.0,
        resize_mode="none",
        density_cache_mode="require",
        density_cache_root=None,
        return_metadata=False,
        **kwargs,
    ):
        super().__init__()

        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.return_metadata = bool(return_metadata)

        if self.split not in self.EXPECTED_SPLITS:
            raise ValueError(
                f"Unsupported split {self.split!r}. "
                f"Expected one of {sorted(self.EXPECTED_SPLITS)}"
            )

        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        if resize_mode != "none":
            raise ValueError(
                "ABD is frozen at native canonical 1024x768. "
                f"resize_mode must be 'none', got {resize_mode!r}"
            )

        if int(out_stride) != 4:
            raise ValueError(
                f"ABD frozen density stride is 4, got {out_stride}"
            )

        if not adaptive:
            raise ValueError(
                "ABD frozen density cache requires adaptive=True"
            )

        expected_parameters = {
            "sigma": (float(sigma), 4.0),
            "beta": (float(beta), 0.3),
            "k_neighbors": (int(k_neighbors), 3),
            "sigma_min": (float(sigma_min), 1.0),
            "sigma_max": (float(sigma_max), 12.0),
        }

        for name, (actual, expected) in expected_parameters.items():
            if actual != expected:
                raise ValueError(
                    f"ABD frozen recipe mismatch: "
                    f"{name}={actual}, expected {expected}"
                )

        if density_cache_mode != "require":
            raise ValueError(
                "ABD training requires the frozen density cache. "
                "Use density_cache_mode='require'."
            )

        manifest_path = Path(split_manifest)

        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path

        manifest_path = manifest_path.resolve()

        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        if density_cache_root is None:
            density_cache_root = (
                Path("density_maps")
                / self.EXPECTED_RECIPE
            )

        density_cache_root = Path(density_cache_root)

        if not density_cache_root.is_absolute():
            density_cache_root = (
                self.root
                / density_cache_root
            )

        self.density_cache_root = (
            density_cache_root.resolve()
        )

        if not self.density_cache_root.is_dir():
            raise FileNotFoundError(
                self.density_cache_root
            )

        metadata_path = (
            self.density_cache_root
            / "metadata.json"
        )

        cache_manifest_path = (
            self.density_cache_root
            / "density_manifest.csv"
        )

        for required in (
            metadata_path,
            cache_manifest_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if metadata.get("dataset") != "ABD":
            raise ValueError(
                "Density-cache dataset identity mismatch: "
                f"{metadata.get('dataset')!r}"
            )

        if (
            metadata.get("recipe_id")
            != self.EXPECTED_RECIPE
        ):
            raise ValueError(
                "Density-cache recipe mismatch: "
                f"{metadata.get('recipe_id')!r}"
            )

        with manifest_path.open(
            newline="",
            encoding="utf-8",
        ) as f:
            benchmark_rows = list(
                csv.DictReader(f)
            )

        with cache_manifest_path.open(
            newline="",
            encoding="utf-8",
        ) as f:
            cache_rows = list(
                csv.DictReader(f)
            )

        cache_by_id = {
            row["sample_id"]: row
            for row in cache_rows
        }

        if len(cache_by_id) != len(cache_rows):
            raise ValueError(
                "Duplicate sample_id in density-cache manifest"
            )

        rows = [
            row
            for row in benchmark_rows
            if row["split"] == self.split
        ]

        expected_n = self.EXPECTED_SPLITS[
            self.split
        ]

        if len(rows) != expected_n:
            raise ValueError(
                f"ABD split {self.split!r} expected "
                f"{expected_n} samples, found {len(rows)}"
            )

        self.samples = []

        for row in rows:
            sample_id = row["sample_id"]

            if sample_id not in cache_by_id:
                raise KeyError(
                    f"Missing density-cache row for {sample_id}"
                )

            cache_row = cache_by_id[
                sample_id
            ]

            if (
                cache_row["experiment_split"]
                != self.split
            ):
                raise ValueError(
                    f"Split mismatch for {sample_id}: "
                    f"benchmark={self.split}, "
                    f"cache={cache_row['experiment_split']}"
                )

            image_path = (
                self.root
                / row["image_file"]
            ).resolve()

            points_path = (
                self.root
                / row["points_file"]
            ).resolve()

            density_path = (
                self.root
                / cache_row["density_path"]
            ).resolve()

            for required in (
                image_path,
                points_path,
                density_path,
            ):
                if not required.is_file():
                    raise FileNotFoundError(
                        required
                    )

            self.samples.append(
                {
                    "sample_id":
                        sample_id,

                    "dataset_source":
                        row["dataset_source"],

                    "source_id":
                        row["source_id"],

                    "original_id":
                        row["original_id"],

                    "split":
                        self.split,

                    "count":
                        int(row["count"]),

                    "image_path":
                        image_path,

                    "points_path":
                        points_path,

                    "density_path":
                        density_path,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(
            str(sample["image_path"]),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Unreadable image: "
                f"{sample['image_path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        h, w = image.shape[:2]

        if (
            h,
            w,
        ) != self.EXPECTED_IMAGE_HW:
            raise ValueError(
                f"{sample['sample_id']}: "
                f"unexpected image shape {(h, w)}"
            )

        density = np.asarray(
            np.load(
                sample["density_path"],
                allow_pickle=False,
            ),
            dtype=np.float32,
        )

        if density.shape != self.EXPECTED_DENSITY_HW:
            raise ValueError(
                f"{sample['sample_id']}: "
                f"unexpected density shape "
                f"{density.shape}"
            )

        expected_count = sample[
            "count"
        ]

        mass_error = abs(
            float(
                density.sum(
                    dtype=np.float64
                )
            )
            - expected_count
        )

        if mass_error > 1e-4:
            raise ValueError(
                f"{sample['sample_id']}: "
                f"density mass mismatch "
                f"{mass_error:.3e}"
            )

        image = np.ascontiguousarray(
            image.transpose(2, 0, 1)
        )

        image_tensor = (
            torch.from_numpy(image)
            .float()
            .div_(255.0)
        )

        density_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    density[None, ...]
                )
            )
            .float()
        )

        if not self.return_metadata:
            return (
                image_tensor,
                density_tensor,
            )

        metadata = {
            "sample_id":
                sample["sample_id"],

            "dataset_source":
                sample["dataset_source"],

            "source_id":
                sample["source_id"],

            "original_id":
                sample["original_id"],

            "split":
                sample["split"],

            "count":
                sample["count"],
        }

        return (
            image_tensor,
            density_tensor,
            metadata,
        )

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .density_map import make_density_map


class ArroyoWasteDataset(Dataset):
    """
    ArroyoWaste full-resolution dataset.

    Canonical benchmark contract
    ----------------------------
    train : all stored variants belonging to train sources
    val   : base (000000) samples only
    test  : base (000000) samples only

    Returns
    -------
    image_tensor : torch.float32 [3, H, W]
    density      : torch.float32 [1, H/out_stride, W/out_stride]

    With return_metadata=True:
        (image_tensor, density_tensor, metadata)
    """

    VALID_SPLITS = {"train", "val", "test"}
    VALID_CACHE_MODES = {"off", "prefer", "require"}

    def __init__(
        self,
        root,
        split,
        split_manifest=None,
        out_stride=4,
        sigma=4.0,
        adaptive=True,
        beta=0.3,
        k_neighbors=3,
        sigma_min=1.0,
        sigma_max=12.0,
        density_cache_mode="off",
        density_cache_root=None,
        return_metadata=False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)

        self.out_stride = int(out_stride)

        self.sigma = float(sigma)
        self.adaptive = bool(adaptive)
        self.beta = float(beta)
        self.k_neighbors = int(k_neighbors)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.density_cache_mode = str(
            density_cache_mode
        )

        self.return_metadata = bool(
            return_metadata
        )

        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {self.split}"
            )

        if (
            self.density_cache_mode
            not in self.VALID_CACHE_MODES
        ):
            raise ValueError(
                "Invalid density_cache_mode: "
                f"{self.density_cache_mode}"
            )

        if not self.root.exists():
            raise FileNotFoundError(
                self.root
            )

        # --------------------------------------------------
        # Frozen benchmark manifest
        # --------------------------------------------------

        if split_manifest is None:
            split_manifest = (
                self.root
                / "manifests"
                / "benchmark_seed7.csv"
            )
        else:
            split_manifest = Path(
                split_manifest
            ).expanduser()

            if not split_manifest.is_absolute():
                split_manifest = (
                    self.root
                    / split_manifest
                )

        self.split_manifest = (
            split_manifest.resolve()
        )

        if not self.split_manifest.exists():
            raise FileNotFoundError(
                self.split_manifest
            )

        # --------------------------------------------------
        # Density cache
        # --------------------------------------------------

        if density_cache_root is None:
            self.density_cache_root = None

        else:
            density_cache_root = Path(
                density_cache_root
            ).expanduser()

            if not density_cache_root.is_absolute():
                density_cache_root = (
                    self.root
                    / density_cache_root
                )

            self.density_cache_root = (
                density_cache_root.resolve()
            )

        if (
            self.density_cache_mode == "require"
            and self.density_cache_root is None
        ):
            raise ValueError(
                "density_cache_root is required "
                "when density_cache_mode='require'."
            )

        self.cache_manifest_by_id = {}
        self.cache_metadata = None

        if self.density_cache_root is not None:
            self._load_density_cache_metadata()

        self.samples = self._load_samples()

        if not self.samples:
            raise RuntimeError(
                f"No samples for split={self.split}"
            )

        if self.density_cache_mode == "require":
            self._validate_required_cache_coverage()

    @staticmethod
    def _sha256(path):
        h = hashlib.sha256()

        with Path(path).open("rb") as stream:
            for chunk in iter(
                lambda: stream.read(
                    1024 * 1024
                ),
                b"",
            ):
                h.update(chunk)

        return h.hexdigest()

    def _load_density_cache_metadata(self):

        metadata_path = (
            self.density_cache_root
            / "metadata.json"
        )

        manifest_path = (
            self.density_cache_root
            / "manifest.csv"
        )

        if not metadata_path.exists():

            if self.density_cache_mode == "require":
                raise FileNotFoundError(
                    metadata_path
                )

            return

        if not manifest_path.exists():

            if self.density_cache_mode == "require":
                raise FileNotFoundError(
                    manifest_path
                )

            return

        self.cache_metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        # ----------------------------------------------
        # Dataset identity
        # ----------------------------------------------

        dataset_name = (
            self.cache_metadata.get(
                "dataset"
            )
        )

        if dataset_name != "ArroyoWaste":
            raise ValueError(
                "Density cache dataset mismatch: "
                f"{dataset_name!r} != "
                "'ArroyoWaste'"
            )

        # ----------------------------------------------
        # Frozen benchmark provenance
        # ----------------------------------------------

        expected_manifest_sha = (
            self.cache_metadata.get(
                "benchmark_manifest_sha256"
            )
        )

        actual_manifest_sha = (
            self._sha256(
                self.split_manifest
            )
        )

        if (
            expected_manifest_sha
            and expected_manifest_sha
            != actual_manifest_sha
        ):
            raise ValueError(
                "Density cache was generated from "
                "a different benchmark manifest: "
                f"cache={expected_manifest_sha}, "
                f"current={actual_manifest_sha}"
            )

        # ----------------------------------------------
        # Density recipe
        # ----------------------------------------------

        parameters = (
            self.cache_metadata.get(
                "parameters",
                {},
            )
        )

        expected_parameters = {
            "resize_mode":
                "none",

            "out_stride":
                self.out_stride,

            "adaptive":
                self.adaptive,

            "sigma_fallback":
                self.sigma,

            "beta":
                self.beta,

            "k_neighbors":
                self.k_neighbors,

            "sigma_min":
                self.sigma_min,

            "sigma_max":
                self.sigma_max,
        }

        for key, expected_value in (
            expected_parameters.items()
        ):
            actual_value = parameters.get(
                key
            )

            if actual_value != expected_value:
                raise ValueError(
                    "Density cache recipe mismatch "
                    f"for {key}: "
                    f"cache={actual_value}, "
                    f"dataset={expected_value}"
                )

        # ----------------------------------------------
        # Cache manifest
        # ----------------------------------------------

        with manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:

            reader = csv.DictReader(
                stream
            )

            required = {
                "sample_id",
                "source_id",
                "experiment_split",
                "image_path",
                "points_path",
                "density_path",
                "count",
            }

            missing = (
                required
                - set(
                    reader.fieldnames
                    or []
                )
            )

            if missing:
                raise ValueError(
                    "Density cache manifest "
                    "missing columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                sample_id = row[
                    "sample_id"
                ]

                if (
                    sample_id
                    in self.cache_manifest_by_id
                ):
                    raise ValueError(
                        "Duplicate cache sample_id: "
                        f"{sample_id}"
                    )

                self.cache_manifest_by_id[
                    sample_id
                ] = row

    def _load_samples(self):

        samples = []

        with self.split_manifest.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:

            reader = csv.DictReader(
                stream
            )

            required = {
                "sample_id",
                "source_id",
                "legacy_parent_id",
                "augmentation_prefix",
                "augmentation",
                "is_base",
                "image_file",
                "points_file",
                "width",
                "height",
                "count",
                "split",
            }

            missing = (
                required
                - set(
                    reader.fieldnames
                    or []
                )
            )

            if missing:
                raise ValueError(
                    "Benchmark manifest missing "
                    "columns: "
                    f"{sorted(missing)}"
                )

            for row in reader:

                if row["split"] != self.split:
                    continue

                sample_id = row[
                    "sample_id"
                ]

                image_path = (
                    self.root
                    / row["image_file"]
                )

                points_path = (
                    self.root
                    / row["points_file"]
                )

                samples.append(
                    {
                        "global_id":
                            sample_id,

                        "sample_id":
                            sample_id,

                        "source_id":
                            row[
                                "source_id"
                            ],

                        "legacy_parent_id":
                            row[
                                "legacy_parent_id"
                            ],

                        "augmentation_prefix":
                            row[
                                "augmentation_prefix"
                            ],

                        "augmentation":
                            row[
                                "augmentation"
                            ],

                        "is_base":
                            bool(
                                int(
                                    row[
                                        "is_base"
                                    ]
                                )
                            ),

                        "manifest_count":
                            int(
                                row["count"]
                            ),

                        "manifest_width":
                            int(
                                row["width"]
                            ),

                        "manifest_height":
                            int(
                                row["height"]
                            ),

                        "image_path":
                            image_path,

                        "points_path":
                            points_path,
                    }
                )

        samples.sort(
            key=lambda x:
                x["global_id"]
        )

        return samples

    def _validate_required_cache_coverage(self):

        missing = []

        for sample in self.samples:

            sample_id = sample[
                "sample_id"
            ]

            row = (
                self.cache_manifest_by_id.get(
                    sample_id
                )
            )

            if row is None:
                missing.append(
                    sample_id
                )
                continue

            if (
                row["experiment_split"]
                != self.split
            ):
                raise ValueError(
                    f"{sample_id}: cache split "
                    f"{row['experiment_split']} "
                    f"!= dataset split "
                    f"{self.split}"
                )

            if (
                int(row["count"])
                != sample[
                    "manifest_count"
                ]
            ):
                raise ValueError(
                    f"{sample_id}: cache count "
                    "does not match benchmark "
                    "manifest."
                )

            density_path = (
                self.root
                / row["density_path"]
            )

            if not density_path.exists():
                missing.append(
                    sample_id
                )

        if missing:
            raise RuntimeError(
                "Required density cache is "
                "incomplete. Missing "
                f"{len(missing)} samples. "
                f"Examples: {missing[:10]}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_points(
        self,
        sample,
    ):

        if not sample[
            "points_path"
        ].exists():
            raise FileNotFoundError(
                sample["points_path"]
            )

        points = np.asarray(
            np.load(
                sample["points_path"],
                allow_pickle=False,
            ),
            dtype=np.float32,
        ).reshape(-1, 2)

        if (
            len(points)
            != sample[
                "manifest_count"
            ]
        ):
            raise ValueError(
                "Point count mismatch for "
                f"{sample['sample_id']}: "
                f"{len(points)} != "
                f"{sample['manifest_count']}"
            )

        if not np.isfinite(
            points
        ).all():
            raise ValueError(
                "Points contain NaN/Inf: "
                f"{sample['sample_id']}"
            )

        return points

    @staticmethod
    def _pad_to_multiple(
        image,
        multiple,
    ):

        h, w = image.shape[:2]

        pad_h = (
            multiple
            - h % multiple
        ) % multiple

        pad_w = (
            multiple
            - w % multiple
        ) % multiple

        if (
            pad_h == 0
            and pad_w == 0
        ):
            return (
                image,
                0,
                0,
            )

        image = cv2.copyMakeBorder(
            image,
            0,
            pad_h,
            0,
            pad_w,
            borderType=(
                cv2.BORDER_CONSTANT
            ),
            value=(0, 0, 0),
        )

        return (
            image,
            pad_h,
            pad_w,
        )

    def _load_cached_density(
        self,
        sample,
        expected_shape,
    ):

        row = (
            self.cache_manifest_by_id.get(
                sample["sample_id"]
            )
        )

        if row is None:

            if (
                self.density_cache_mode
                == "require"
            ):
                raise KeyError(
                    "Missing cache entry: "
                    f"{sample['sample_id']}"
                )

            return None

        density_path = (
            self.root
            / row["density_path"]
        )

        if not density_path.exists():

            if (
                self.density_cache_mode
                == "require"
            ):
                raise FileNotFoundError(
                    density_path
                )

            return None

        density = np.asarray(
            np.load(
                density_path,
                allow_pickle=False,
            ),
            dtype=np.float32,
        )

        if (
            density.shape
            != expected_shape
        ):
            raise ValueError(
                "Cached density shape mismatch "
                f"for {sample['sample_id']}: "
                f"{density.shape} != "
                f"{expected_shape}"
            )

        if not np.isfinite(
            density
        ).all():
            raise ValueError(
                "Cached density contains "
                "NaN/Inf: "
                f"{sample['sample_id']}"
            )

        return density

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[
            index
        ]

        if not sample[
            "image_path"
        ].exists():
            raise FileNotFoundError(
                sample["image_path"]
            )

        image = cv2.imread(
            str(
                sample[
                    "image_path"
                ]
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                "Unreadable image: "
                f"{sample['image_path']}"
            )

        original_h, original_w = (
            image.shape[:2]
        )

        if (
            original_w
            != sample[
                "manifest_width"
            ]
            or original_h
            != sample[
                "manifest_height"
            ]
        ):
            raise ValueError(
                f"{sample['sample_id']}: "
                f"image={original_w}x"
                f"{original_h}, "
                "manifest="
                f"{sample['manifest_width']}x"
                f"{sample['manifest_height']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        points = self._load_points(
            sample
        )

        image, pad_h, pad_w = (
            self._pad_to_multiple(
                image,
                self.out_stride,
            )
        )

        processed_h, processed_w = (
            image.shape[:2]
        )

        expected_density_shape = (
            processed_h
            // self.out_stride,
            processed_w
            // self.out_stride,
        )

        density = None
        density_source = "online"

        if (
            self.density_cache_mode
            != "off"
        ):
            density = (
                self._load_cached_density(
                    sample,
                    expected_density_shape,
                )
            )

            if density is not None:
                density_source = "cache"

        if density is None:

            density = make_density_map(
                points=points,
                height=processed_h,
                width=processed_w,
                sigma=self.sigma,
                out_stride=(
                    self.out_stride
                ),
                adaptive=self.adaptive,
                beta=self.beta,
                k_neighbors=(
                    self.k_neighbors
                ),
                sigma_min=(
                    self.sigma_min
                ),
                sigma_max=(
                    self.sigma_max
                ),
            )

        density = np.asarray(
            density,
            dtype=np.float32,
        )

        if (
            density.shape
            != expected_density_shape
        ):
            raise ValueError(
                f"{sample['sample_id']}: "
                "density shape mismatch "
                f"{density.shape} != "
                f"{expected_density_shape}"
            )

        if not np.isfinite(
            density
        ).all():
            raise ValueError(
                "Density contains NaN/Inf: "
                f"{sample['sample_id']}"
            )

        density_sum = float(
            density.sum(
                dtype=np.float64
            )
        )

        count_error = abs(
            density_sum
            - len(points)
        )

        tolerance = max(
            1e-4,
            len(points) * 1e-6,
        )

        if count_error > tolerance:
            raise ValueError(
                "Density mass mismatch for "
                f"{sample['sample_id']}: "
                f"sum={density_sum}, "
                f"count={len(points)}, "
                f"error={count_error}"
            )

        image_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    image.astype(
                        np.float32
                    )
                    / 255.0
                )
            )
            .permute(
                2,
                0,
                1,
            )
            .contiguous()
        )

        density_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    density
                )
            )
            .unsqueeze(0)
            .contiguous()
        )

        if not self.return_metadata:
            return (
                image_tensor,
                density_tensor,
            )

        metadata = {
            "global_id":
                sample[
                    "global_id"
                ],

            "sample_id":
                sample[
                    "sample_id"
                ],

            "source_id":
                sample[
                    "source_id"
                ],

            "legacy_parent_id":
                sample[
                    "legacy_parent_id"
                ],

            "augmentation_prefix":
                sample[
                    "augmentation_prefix"
                ],

            "augmentation":
                sample[
                    "augmentation"
                ],

            "is_base":
                sample[
                    "is_base"
                ],

            "dataset":
                "ArroyoWaste",

            "split":
                self.split,

            "count":
                len(points),

            "density_source":
                density_source,

            "original_size": (
                original_h,
                original_w,
            ),

            "processed_size": (
                processed_h,
                processed_w,
            ),

            "padding": (
                pad_h,
                pad_w,
            ),

            "scale": (
                1.0,
                1.0,
            ),
        }

        return (
            image_tensor,
            density_tensor,
            metadata,
        )

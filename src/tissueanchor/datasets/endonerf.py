"""EndoNeRF sample adapter for the existing shared query NeRF v1 checkpoint.

Only 48 non-test frames supply the frozen OpenD4RT query cache. The remaining
training frames can supervise a scene-local 100-step adaptation. Held-out RGB
and tissue masks are loaded only for rendering/evaluation; test depth is never
opened. Released depth is an 8-bit stereo estimate, not metric ground truth.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tissueanchor.datasets.stereomis import SharedScene


class EndoNeRFSharedScene:
    def __init__(self, name: str, include_adaptation_targets: bool = False,
                 load_test_rgb: bool = True, *, root):
        self.name = "EndoNeRF_" + name
        self.root = Path(root) / name
        paths = sorted((self.root / "images").glob("*.png"))
        if not paths:
            raise FileNotFoundError(self.root / "images")
        n = len(paths)
        mask_paths = sorted((self.root / "masks").glob("*.png"))
        depth_paths = sorted((self.root / "depth").glob("*.png"))
        if len(mask_paths) != n or len(depth_paths) != n:
            raise ValueError("EndoNeRF RGB/mask/depth frame counts differ")
        self.original_ids = [p.stem for p in paths]
        def numeric_ids(items):
            return [int(re.findall(r"\d+", item.stem)[-1]) for item in items]

        if numeric_ids(mask_paths) != numeric_ids(paths):
            raise ValueError("EndoNeRF mask frame IDs differ")
        if numeric_ids(depth_paths) != numeric_ids(paths):
            raise ValueError("EndoNeRF depth frame IDs differ")

        train = [i for i in range(n) if (i - 1) % 8 != 0]
        test = [i for i in range(n) if (i - 1) % 8 == 0]
        source = np.asarray(train)[np.linspace(0, len(train) - 1, 48).round().astype(int)].tolist()
        if len(set(source)) != 48 or set(source) & set(test):
            raise ValueError("Invalid 48-frame context/test split")
        remaining = [i for i in train if i not in set(source)]
        self.original_positions = source + test
        self.context = list(range(48))
        self.test = list(range(48, 48 + len(test)))
        self.targets: list[int] = []
        self.adaptation_original_positions = remaining if include_adaptation_targets else []
        self.original_train_count = len(train)
        self.original_test_count = len(test)
        self.original_frame_count = n
        self.h, self.w = np.asarray(Image.open(paths[0]).convert("RGB")).shape[:2]

        pose_raw = np.load(self.root / "poses_bounds.npy")
        if len(pose_raw) != n:
            raise ValueError("EndoNeRF pose/frame count mismatch")
        pose_data = pose_raw[:, :-2].reshape(n, 3, 5)
        focal = float(pose_data[0, 2, 4])
        self.K = torch.tensor([[focal, 0., self.w / 2],
                               [0., focal, self.h / 2],
                               [0., 0., 1.]], dtype=torch.float32)
        all_pose = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        all_pose[:, :3, :4] = pose_data[:, :, :4]
        self.times = torch.arange(n, dtype=torch.float32)[self.original_positions] / n
        self.meta = {"frame_ids": [self.original_ids[i] for i in self.original_positions]}

        def load_image(index: int) -> torch.Tensor:
            return torch.from_numpy(np.asarray(Image.open(paths[index]).convert("RGB")).copy())

        def load_mask(index: int) -> torch.Tensor:
            return torch.from_numpy((np.asarray(Image.open(mask_paths[index]).convert("L")) < 128).copy())

        def load_prior(index: int) -> torch.Tensor:
            raw = np.asarray(Image.open(depth_paths[index]), dtype=np.float32)
            positive = raw[raw > 0]
            if len(positive) == 0:
                raise ValueError(f"No positive released depth: {depth_paths[index]}")
            lo, hi = np.percentile(positive, (3.0, 99.8))
            return torch.from_numpy(np.clip(raw, lo, hi).astype(np.float32))

        self.images = torch.stack([load_image(i) if slot < 48 or load_test_rgb
                                   else torch.zeros(self.h, self.w, 3, dtype=torch.uint8)
                                   for slot, i in enumerate(self.original_positions)])
        self.masks = torch.stack([load_mask(i) if slot < 48 or load_test_rgb
                                  else torch.zeros(self.h, self.w, dtype=torch.bool)
                                  for slot, i in enumerate(self.original_positions)])
        depth = torch.stack([load_prior(i) if slot < 48 else torch.zeros(self.h, self.w)
                             for slot, i in enumerate(self.original_positions)])
        values = depth[:48][self.masks[:48] & (depth[:48] > 0)]
        if not len(values):
            raise ValueError("No valid source tissue depth")
        self.scale_cm = float(values.median())  # Interface name; EndoNeRF units are not centimetres.
        self.depth_output_folder = "depth_prior_units"
        self.depths = depth / self.scale_cm
        self.depth_valid = depth > 0
        values_np = values.numpy().reshape(-1)
        if len(values_np) > 1_000_000:
            values_np = values_np[np.linspace(0, len(values_np) - 1, 1_000_000, dtype=np.int64)]
        q005, q995 = np.quantile(values_np, (.005, .995))
        self.near = max(.02, float(q005) * .65 / self.scale_cm)
        self.far = float(q995) * 1.35 / self.scale_cm
        poses = all_pose[self.original_positions].copy()
        poses[:, :3, 3] /= self.scale_cm
        self.c2w = torch.from_numpy(poses)
        self.w2c = torch.linalg.inv(self.c2w)
        self.context_times = self.times[self.context].contiguous()
        self.pixels = [torch.nonzero(mask, as_tuple=False) for mask in self.masks]
        self._gpu_context = None

        if include_adaptation_targets and remaining:
            start = len(self.images)
            self.images = torch.cat((self.images, torch.stack([load_image(i) for i in remaining])))
            self.masks = torch.cat((self.masks, torch.stack([load_mask(i) for i in remaining])))
            target_depth = torch.stack([load_prior(i) for i in remaining])
            self.depths = torch.cat((self.depths, target_depth / self.scale_cm))
            self.depth_valid = torch.cat((self.depth_valid, target_depth > 0))
            self.times = torch.cat((self.times, torch.tensor(remaining, dtype=torch.float32) / n))
            target_poses = all_pose[remaining].copy()
            target_poses[:, :3, 3] /= self.scale_cm
            self.c2w = torch.cat((self.c2w, torch.from_numpy(target_poses)))
            self.w2c = torch.linalg.inv(self.c2w)
            self.pixels.extend(torch.nonzero(mask, as_tuple=False) for mask in self.masks[start:])
            self.targets = list(range(start, start + len(remaining)))

    context_tensors = SharedScene.context_tensors
    rays = SharedScene.rays

    def manifest(self) -> dict:
        return {
            "scene": self.name, "dataset": "EndoNeRF official sample",
            "source": str(self.root), "original_frame_count": self.original_frame_count,
            "original_train_count": self.original_train_count,
            "original_test_count": self.original_test_count,
            "original_source_positions": self.original_positions[:48],
            "original_evaluation_positions": self.original_positions[48:],
            "frame_ids": self.meta["frame_ids"],
            "context_indices": self.context, "test_indices": self.test,
            "image_wh": [self.w, self.h], "K": self.K.tolist(),
            "pose_sha256": hashlib.sha256((self.root / "poses_bounds.npy").read_bytes()).hexdigest(),
            "source_depth_scale_in_prior_units": self.scale_cm,
            "depth_units": "released 8-bit stereo-prior units, not metric centimetres",
            "near": self.near, "far": self.far,
            "mask": "released mask < 128 is valid tissue",
            "source_depth_preprocessing": "per-frame 3rd/99.8th percentile clipping, matching TMI loader",
            "heldout_RGB_in_conditioning": False,
            "heldout_depth_loaded": False,
            "context_target_disjoint": True,
        }

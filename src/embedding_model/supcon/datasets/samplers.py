"""Sampling and collation helpers for SupCon datasets."""
import random
from bisect import bisect_left
from itertools import accumulate
from typing import List

import torch
from torch.utils.data import Dataset, Sampler


class IndexWithScale:
    """包装索引和尺度的辅助类"""

    def __init__(self, idx: int, image_size: int):
        self.idx = idx
        self.image_size = image_size


class MultiScaleBatchSampler(Sampler):
    """
    多尺度Batch采样器。

    确保每个 batch 内的所有样本使用相同图像尺度；DDP 下按 rank/world_size
    将完整 batch 列表切分到各进程。
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        image_sizes: List[int],
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_sizes = image_sizes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.epoch)
        indices = self.indices.copy()
        if self.shuffle:
            rng.shuffle(indices)

        all_batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            image_size = rng.choice(self.image_sizes)
            all_batches.append([IndexWithScale(idx, image_size) for idx in batch_indices])

        for i in range(self.rank, len(all_batches), self.world_size):
            yield all_batches[i]

    def __len__(self):
        total = (len(self.dataset) // self.batch_size if self.drop_last
                 else (len(self.dataset) + self.batch_size - 1) // self.batch_size)
        return (total + self.world_size - 1) // self.world_size


class SceneBatchSampler(Sampler):
    """
    Step-based sampler for multi-scene SupCon training.

    每个 step 先按 scene 权重采样一个 scene，再从该 scene 中采一个 batch。
    DDP 下 scene_idx 和 image_size 由 rank-independent step seed 决定，所有
    rank 同一步保持一致；样本索引按 world_size 切片，样本不足时自动 replacement。
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        image_sizes: List[int],
        total_steps: int,
        scene_sampling: str = "size_temperature",
        scene_sampling_alpha: float = 0.5,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        start_step: int = 1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须 > 0，当前为 {total_steps}")
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，当前为 {batch_size}")
        if not image_sizes:
            raise ValueError("image_sizes 不能为空")
        if not hasattr(dataset, "scene_global_indices"):
            raise TypeError("SceneBatchSampler 需要 dataset 暴露 scene_global_indices")

        self.dataset = dataset
        self.batch_size = batch_size
        self.image_sizes = image_sizes
        self.total_steps = int(total_steps)
        self.scene_sampling = scene_sampling
        self.scene_sampling_alpha = float(scene_sampling_alpha)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = int(start_step)
        self.scene_indices = dataset.scene_global_indices
        self.scene_weights = self._build_scene_weights()
        self.cumulative_weights = list(accumulate(self.scene_weights))

    def set_start_step(self, start_step: int):
        self.start_step = int(start_step)

    def __iter__(self):
        for step in range(self.start_step, self.total_steps + 1):
            step_rng = random.Random(self.seed + step * 1_000_003)
            scene_idx = self._sample_scene(step_rng)
            image_size = step_rng.choice(self.image_sizes)

            pool = self.scene_indices[scene_idx]
            needed = self.batch_size * self.world_size
            if len(pool) >= needed:
                selected = step_rng.sample(pool, needed)
            else:
                selected = [step_rng.choice(pool) for _ in range(needed)]

            begin = self.rank * self.batch_size
            end = begin + self.batch_size
            yield [IndexWithScale(idx, image_size) for idx in selected[begin:end]]

    def __len__(self):
        return max(0, self.total_steps - self.start_step + 1)

    def _build_scene_weights(self) -> List[float]:
        counts = [len(indices) for indices in self.scene_indices]
        if not counts or any(count <= 0 for count in counts):
            raise ValueError(f"SceneBatchSampler 只接受非空 scene，当前 counts={counts}")

        if self.scene_sampling == "uniform":
            return [1.0] * len(counts)
        if self.scene_sampling in {"size", "sample_count"}:
            return [float(count) for count in counts]
        if self.scene_sampling == "size_temperature":
            alpha = max(0.0, min(1.0, self.scene_sampling_alpha))
            return [float(count) ** alpha for count in counts]
        raise ValueError(
            "scene_sampling 仅支持 uniform / size / sample_count / size_temperature，"
            f"当前为 {self.scene_sampling!r}"
        )

    def _sample_scene(self, rng: random.Random) -> int:
        threshold = rng.random() * self.cumulative_weights[-1]
        return bisect_left(self.cumulative_weights, threshold)


def multi_scale_collate_fn(batch_data):
    """Collate samples produced by ``MultiScaleBatchSampler``."""
    result = {}
    for key in batch_data[0].keys():
        if key in ['view1_image', 'view1_mask', 'view2_image', 'view2_mask']:
            result[key] = torch.stack([item[key] for item in batch_data])
        elif key in ['label', 'scene_idx']:
            result[key] = torch.tensor([item[key] for item in batch_data])
        else:
            result[key] = [item[key] for item in batch_data]
    return result

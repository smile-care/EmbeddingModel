"""Sampling and collation helpers for SupCon datasets."""
import random
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


def multi_scale_collate_fn(batch_data):
    """Collate samples produced by ``MultiScaleBatchSampler``."""
    result = {}
    for key in batch_data[0].keys():
        if key in ['view1_image', 'view1_mask', 'view2_image', 'view2_mask']:
            result[key] = torch.stack([item[key] for item in batch_data])
        elif key in ['label']:
            result[key] = torch.tensor([item[key] for item in batch_data])
        else:
            result[key] = [item[key] for item in batch_data]
    return result

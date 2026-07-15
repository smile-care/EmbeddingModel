"""Deterministic step-based uniform sampling over the merged sample pool."""

from __future__ import annotations

import random

from torch.utils.data import Sampler


class UniformStepBatchSampler(Sampler[list[int]]):
    """Draw every global batch uniformly without scene or label weighting."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        total_steps: int,
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        start_step: int = 1,
    ) -> None:
        if dataset_size <= 0 or batch_size <= 0 or total_steps <= 0:
            raise ValueError("dataset_size、batch_size 和 total_steps 必须 > 0")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_step = int(start_step)

    def set_start_step(self, start_step: int) -> None:
        self.start_step = int(start_step)

    def __iter__(self):
        global_batch_size = self.batch_size * self.world_size
        population = range(self.dataset_size)
        for step in range(self.start_step, self.total_steps + 1):
            generator = random.Random(self.seed + step * 1_000_003)
            if self.dataset_size >= global_batch_size:
                global_indices = generator.sample(population, global_batch_size)
            else:
                global_indices = [generator.randrange(self.dataset_size) for _ in range(global_batch_size)]
            begin = self.rank * self.batch_size
            yield global_indices[begin : begin + self.batch_size]

    def __len__(self) -> int:
        return max(0, self.total_steps - self.start_step + 1)

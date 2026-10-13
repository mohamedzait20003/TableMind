"""
Resumable shuffling sampler.

Replays the same shuffled order for a given (seed, epoch) and can start part
way through it, so training resumed from a mid-epoch checkpoint continues
with exactly the samples it had not seen yet.
"""

import torch
from typing import Iterator
from torch.utils.data import Sampler


class ResumableSampler(Sampler):
    """Shuffles once per epoch from `seed + epoch`; skips the first `start` samples."""

    def __init__(self, num_samples: int, seed: int = 42):
        """
        Args:
            num_samples: Dataset size
            seed: Base seed of the per-epoch permutation
        """
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        """Select the epoch's permutation and the first sample to yield."""
        self.epoch = epoch
        self.start = min(start, self.num_samples)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.num_samples, generator=generator).tolist()
        return iter(order[self.start:])

    def __len__(self) -> int:
        return self.num_samples - self.start

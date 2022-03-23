"""Utility module for model subpackage."""
from typing import Sequence

import torch

from comic_cover_generator.ml.dataset import CoverDatasetItem
from comic_cover_generator.typing import TypedDict


class CoverDatasetBatch(TypedDict):
    """Cover dataset batch item."""

    image: torch.Tensor
    title_seq: Sequence[torch.Tensor]


def collate_fn(batch: Sequence[CoverDatasetItem]) -> CoverDatasetBatch:
    """Collate function for batch conditioning in dataloader.

    Args:
        batch (Sequence[CoverDatasetItem]): List of dictionaries in the batch.

    Returns:
        CoverDatasetBatch: Dictionary of sequences in batch.
    """
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}

    batch["image"] = torch.stack([img for img in batch["image"]])
    batch["title_seq"] = [seq for seq in batch["title_seq"]]

    return batch

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
    batch = {k: [dic[k] for dic in batch] for k in ["image", "title_seq"]}

    batch["image"] = torch.stack([img for img in batch["image"]])
    batch["title_seq"] = [seq for seq in batch["title_seq"]]

    return batch


def make_captioned_grid(
    images: torch.Tensor,
    captions: Sequence[str],
    grid_shape=(4, 4),
    value_range=(-1, 1),
    normalize=True,
    fname: str = None,
    figsize=(20, 20),
    fontsize=9,
) -> None:
    """Make a captioned grid using matploblib imshow.

    Args:
        images (torch.Tensor): Input images. Should have 4 dimensions and be channel first.
        captions (Sequence[str]): Input captions.
        grid_shape (tuple, optional): Defaults to (4, 4).
        value_range (tuple, optional): Defaults to (-1, 1).
        normalize (bool, optional): Defaults to True.
        fname (str, optional): Defaults to None.
        figsize (tuple, optional): Defaults to (20, 20).
        fontsize (int, optional): Defaults to 9.
    """
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision.transforms.functional import to_pil_image

    if normalize is True:
        images = (images - value_range[0]) / (value_range[1] - value_range[0])

    images = [np.asarray(to_pil_image(img)) for img in images]

    fig, axes = plt.subplots(
        nrows=grid_shape[0],
        ncols=grid_shape[1],
        figsize=figsize,
        constrained_layout=True,
    )
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx])
        ax.set_title(
            textwrap.fill(captions[idx], 30), fontsize=fontsize, wrap=True, pad=0
        )
        ax.axis("off")

    if fname is not None:
        plt.savefig(f"{fname}.png")
    else:
        plt.show()
    plt.close()

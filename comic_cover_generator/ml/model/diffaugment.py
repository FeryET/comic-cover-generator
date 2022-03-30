"""Differentiable Augmentations."""
# Copied from: https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py

# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import enum
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F


def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    """Apply random brightness augmentation.

    Args:
        x (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    """Apply random saturation augmentation.

    Args:
        x (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2
    ) + x_mean
    return x


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    """Apply random contrast augmentation.

    Args:
        x (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (
        torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5
    ) + x_mean
    return x


def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Apply rand_translation augmentaion.

    Args:
        x (torch.Tensor): input tensor.
        ratio (float, optional): translation ratio. Defaults to 0.125.

    Returns:
        torch.Tensor:
    """
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    return x


# @torch.jit.script
def rand_cutout(x: torch.Tensor, ratio: float = 0.25) -> torch.Tensor:
    """Apply random cutout augmentaion.

    Args:
        x (torch.Tensor): input tensor.
        ratio (float, optional): Cutout ratio. Defaults to 0.25.

    Returns:
        torch.Tensor:
    """
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


class AugmentPolicy(enum.Enum):
    """Augment policy type."""

    color: Sequence[Callable] = [rand_brightness, rand_saturation, rand_contrast]
    translation: Sequence[Callable] = [rand_translation]
    cutout: Sequence[Callable] = [rand_cutout]


def diff_augment(
    x: torch.Tensor,
    policy: Optional[Sequence[AugmentPolicy]],
    channels_first: bool = True,
) -> torch.Tensor:
    """Apply a differentiable augmentation.

    Args:
        x (torch.Tensor):
        policy (str, optional):
        channels_first (bool, optional):

    Returns:
        torch.Tensor:
    """
    if policy is not None:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy:
            for f in p:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

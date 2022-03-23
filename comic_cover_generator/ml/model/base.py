"""Base Module."""
import torch
from torch import nn

from comic_cover_generator.typing import Protocol


class Freezeable(Protocol):
    """Protocol for freezable nn.Modules."""

    def freeze(self):
        """Freezing method."""
        ...

    def unfreeze(self):
        """Unfreezing method."""
        ...


class ResNetBlock(nn.Module):
    """Resnet block module."""

    def __init__(self, channels: int, p_dropout: float = 0.2) -> None:
        """Initialize a resnet block.

        Args:
            channels (int): channels in resnent block.
            p_dropout (float, optional): Defaults to 0.2.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels // 4, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.InstanceNorm2d(channels // 4, affine=True),
            nn.ReLU(),
            nn.Conv2d(
                channels // 4, channels, kernel_size=1, padding=0, stride=1, bias=False
            ),
        )

        self.relu = nn.Sequential(nn.Dropout2d(p=p_dropout), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of resnet block.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        identity = x
        return self.relu(identity + self.block(x))

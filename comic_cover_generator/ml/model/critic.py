"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import Freezeable
from comic_cover_generator.ml.model.utils import weights_init


class CriticResidualBlock(nn.Module):
    """Residual block used in critic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 2,
        n_blocks: int = 2,
    ) -> None:
        """Residual block for critic.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            expansion_factor (int, optional): Expansion factors for middle channels in residual blocks. Defaults to 2.
            n_blocks (int, optional): Defaults to 3.
        """
        super().__init__()
        mid_channels = in_channels * expansion_factor

        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, mid_channels, kernel_size=3, stride=1, padding=1
                    ),
                    nn.InstanceNorm2d(mid_channels, affine=True),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(
                        mid_channels, in_channels, kernel_size=1, stride=1, padding=0
                    ),
                )
                for _ in range(n_blocks)
            ]
        )
        self.conv.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

        self.downsample = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear"),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block forward pass.

        Args:
            x (torch.Tensor): 4D input.

        Returns:
            torch.Tensor:
        """
        return self.downsample(x) + self.conv(x)


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (64, 64)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        channels = [
            # 64x64
            3,
            # 32x32
            64,
            # 16x16
            128,
            # 8x8
            256,
            # 4x4
            512,
            # 2x2
        ]
        channels = list(zip(channels, channels[1:]))
        self.features = nn.Sequential(
            *[CriticResidualBlock(in_ch, out_ch, 2, 2) for in_ch, out_ch in channels]
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 1),
        )

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.clf(self.features(x))

    def freeze(self):
        """Freeze the critic model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the critic model."""
        for p in self.parameters():
            p.requires_grad = True
        return self

"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import EqualConv2d, EqualLinear, Freezeable
from comic_cover_generator.typing import TypedDict


class CriticResidualBlock(nn.Module):
    """Residual block used in critic."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Residual block for critic.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        self.conv = nn.Sequential(
            EqualConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.1),
            EqualConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.skip = nn.Sequential(
            EqualConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.downsample = nn.Upsample(scale_factor=0.5, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block forward pass.

        Args:
            x (torch.Tensor): 4D input.

        Returns:
            torch.Tensor:
        """
        return self.downsample(self.skip(x) + self.conv(x))


class CriticParams(TypedDict):
    """Critic constructor arguments."""

    channels: Tuple[int, ...]
    input_shape: Tuple[int, int]


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (3, 512, 512, 512, 512),
        input_shape: Tuple[int, int] = (64, 64),
    ) -> None:
        """Initialize a critic module.

        Args:
            channels (Tuple[int, ...], optional): Inputs to the channels used in sequential blocks. Each block will halve the resolution of feature maps after applying the forward method. Defaults to (3, 512, 512, 512, 512).
            input_shape (Tuple[int, int]): Defaults to (64, 64).
        """
        super().__init__()

        self.input_shape = input_shape
        self.channels = channels

        channels = list(zip(channels, channels[1:]))
        self.features = nn.Sequential(
            *[CriticResidualBlock(in_ch, out_ch) for in_ch, out_ch in channels]
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(),
            EqualLinear(self.channels[-1], 1),
        )

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

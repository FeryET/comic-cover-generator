"""Critic Module."""
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.model.base import (
    Blur,
    EqualConv2d,
    EqualLinear,
    Freezeable,
)
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
            EqualConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            EqualConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            BlurDownsample(),
        )

        self.residual = nn.Sequential(
            EqualConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            BlurDownsample(),
        )

        self.register_buffer("residual_scaler", torch.as_tensor([1.0 / math.sqrt(2)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block forward pass.

        Args:
            x (torch.Tensor): 4D input.

        Returns:
            torch.Tensor:
        """
        residual = self.residual(x)
        x = self.conv(x)
        return (x + residual) * self.residual_scaler


class CriticParams(TypedDict):
    """Critic constructor arguments."""

    channels: Tuple[int, ...]
    input_shape: Tuple[int, int]


class MinibatchStdMean(nn.Module):
    """Mini batch std mean layer from ProGAN paper."""

    def __init__(self, eps: float = None) -> None:
        """Initializes the minibatch std mean layer.

        Args:
            eps (float, optional): Defaults to None.
        """
        super().__init__()
        if eps is None:
            self.eps = Constants.eps
        else:
            self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a minibatch std mean for diversity checking in discriminator.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        f_std = torch.sqrt(x.var(dim=0, keepdim=True) + self.eps)
        f_std = f_std.mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat((x, f_std), dim=1)


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
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

        self.from_rgb = nn.Conv2d(3, self.channels[0], 1, 1, 0)

        self.features = nn.Sequential()

        for idx, (in_ch, out_ch) in enumerate(zip(channels, channels[1:]), start=1):
            if idx == len(channels) - 1:
                self.features.append(MinibatchStdMean())
                in_ch = in_ch + 1
            self.features.append(CriticResidualBlock(in_ch, out_ch))

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
        return self.clf(self.features(self.from_rgb(x)))

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


class BlurDownsample(nn.Module):
    """Downsampling layer which smooths and then interpolates the input."""

    def __init__(self):
        """Initialize a downsampling layer."""
        super().__init__()
        # Smoothing layer
        self.blur = Blur()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a downsample.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        # Smoothing or blurring
        x = self.blur(x)
        # Scaled down
        return F.interpolate(
            x, (x.shape[2] // 2, x.shape[3] // 2), mode="bilinear", align_corners=False
        )

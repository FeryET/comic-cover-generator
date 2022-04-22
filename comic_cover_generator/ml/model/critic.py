"""Critic Module."""
import math
from typing import Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

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
        x = (x + residual) * self.residual_scaler
        return x


class CriticParams(TypedDict):
    """Critic constructor arguments."""

    channels: Tuple[int, ...]
    input_shape: Tuple[int, int]


class MinibatchStdMean(nn.Module):
    """Mini batch std mean layer from ProGAN paper."""

    def __init__(self, groups: int = 4, num_channels: int = 1) -> None:
        """Initializes the minibatch std mean layer.

        Args:
            groups (int, optional): Number of groups in the channels. Defaults to 4.
            num_channels (int, optional): Number of channels attached to the output. Defaults to 1.
        """
        super().__init__()
        self.groups = groups
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a minibatch std mean for diversity checking in discriminator.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        b, c, h, w = x.size()
        G = min(b, self.groups) if self.groups is not None else b
        f = self.num_channels
        c = c // f
        y = x.reshape(G, -1, f, c, h, w)
        y = y - y.mean(dim=0)
        y = y.square().mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=(2, 3, 4))
        y = y.reshape(-1, f, 1, 1)
        y = y.repeat(G, 1, h, w)
        x = torch.cat((x, y), dim=1)
        return x


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (64, 128, 256, 512),
        input_shape: Tuple[int, int] = (64, 64),
        mini_batch_std_channels: int = 1,
        mini_batch_std_groups: int = 4,
    ) -> None:
        """Initialize a critic module.

        Args:
            channels (Tuple[int, ...], optional): Inputs to the channels used in sequential blocks. Each block will halve the resolution of feature maps after applying the forward method. Defaults to (3, 512, 512, 512, 512).
            input_shape (Tuple[int, int], optional): Defaults to (64, 64).
            mini_batch_std_channels (int, optional): Defaults to 1.
            mini_batch_std_groups (int, optional): Defaults to 4.
        """
        super().__init__()

        self.input_shape = input_shape
        self.channels = channels

        self.from_rgb = EqualConv2d(3, self.channels[0], 1, 1, 0)

        self.features = nn.Sequential()

        for in_ch, out_ch in zip(channels, channels[1:]):
            self.features.append(CriticResidualBlock(in_ch, out_ch))

        self.clf = nn.Sequential(
            MinibatchStdMean(
                num_channels=mini_batch_std_channels, groups=mini_batch_std_groups
            ),
            EqualConv2d(
                self.channels[-1] + mini_batch_std_channels,
                self.channels[-1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(start_dim=1),
            EqualLinear(self.channels[-1] * 4 * 4, self.channels[-1]),
            nn.LeakyReLU(0.2),
            EqualLinear(self.channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        x = self.from_rgb(x)
        x = self.features(x)
        x = self.clf(x)
        return x

    def create_optimizers(
        self, opt_cls: Type[Optimizer], **optimizer_params
    ) -> Optimizer:
        """Create optimizers for critic.

        Args:
            opt_cls (Type[Optimizer]):

        Returns:
            Optimizer:
        """
        return opt_cls(self.parameters(), **optimizer_params)


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

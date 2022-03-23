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
    """Resnet block layer."""

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

        self.relu = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.Dropout2d(p=p_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward of resnet block.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        identity = x
        return self.relu(identity + self.block(x))


class ResNetScaler(nn.Module):
    """ResNet scaler layer."""

    def __init__(
        self,
        scale_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """Initialize a resnet scaler layer.

        Args:
            scale_type (str): Whether to downsample or upsample. acceptable values are 'up' and 'down'.
            in_channels (int): Number of channels for input.
            out_channels (int): Number of channels for output.
            kernel_size (int): Convolution kernel size.
            stride (int): Convolution stride.
            padding (int): Convolution padding.
        """
        super().__init__()
        if scale_type == "down":
            conv_type = nn.Conv2d
        elif scale_type == "up":
            conv_type = nn.ConvTranspose2d
        else:
            raise KeyError("scale_type can only be either of 'down' or 'up'.")
        self.conv = nn.Sequential(
            conv_type(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call of resnet scaler layer.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.conv(x)

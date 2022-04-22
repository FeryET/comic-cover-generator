"""Base Module."""
import math
from typing import Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.typing import Protocol


class Freezeable(Protocol):
    """Protocol for freezable nn.Modules."""

    def freeze(self):
        """Freezing method."""
        ...

    def unfreeze(self):
        """Unfreezing method."""
        ...


class EqualizedWeight(nn.Module):
    """Equalized learning rate weight module."""

    def __init__(self, *weight_dims) -> None:
        """Initialize an EqualizeWeight module.

        Has a similar interface to torch.rand.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*weight_dims), requires_grad=True)
        self.weight.data.normal_()
        self.register_buffer(
            "scale", torch.as_tensor(math.sqrt(2 / np.prod(weight_dims[1:])))
        )

    def forward(self) -> torch.Tensor:
        """Get equalized weight.

        Returns:
            torch.Tensor: equalized weight.
        """
        return self.weight * self.scale


class EqualLinear(nn.Module):
    """An Equalized Linear Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: Union[bool, float] = True,
    ) -> None:
        """Intiailize an eqaulized linear.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            bias (Union[bool, float], optional): Defaults to True. If not a boolean, it will be the value that initializes the bias.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
        """
        super().__init__()
        self.weight = EqualizedWeight(out_features, in_features)
        if bias is False:
            self.bias = None
        else:
            bias_value = 1 if bias is True else bias
            self.bias = nn.Parameter(
                torch.ones(out_features) * bias_value, requires_grad=True
            )

    def forward(self, input: Tensor) -> Tensor:
        """Forward call of equalized linear.

        Args:
            input (Tensor):

        Returns:
            Tensor:
        """
        return F.linear(
            input,
            self.weight(),
            self.bias,
        )


class EqualConv2d(nn.Module):
    """Equalized Learning Rate 2D Convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """Initializes an equalized conv2d.

        Args:
            in_channels (int): Input feature maps.
            out_channels (int): Output feature maps.
            kernel_size (int): Kernel dims.
            stride (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 0.
            dilation (int, optional): Defaults to 1.
            groups (int, optional): Defaults to 1.
            bias (bool, optional): Defaults to True.
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

        self.weight = EqualizedWeight(
            out_channels, in_channels, kernel_size, kernel_size
        )

        if bias is not False:
            self.bias = nn.Parameter(data=torch.zeros(out_channels), requires_grad=True)
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        """Apply an equalized conv2d.

        Args:
            input (Tensor):

        Returns:
            Tensor:
        """
        return F.conv2d(
            input,
            weight=self.weight(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class ModulatedConv2D(nn.Module):
    """Modulated convolution layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        eps: float = None,
        demod: bool = True,
    ) -> None:
        """Initializes a modulated convolution.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The number of kernel_size.
            padding (int): The number of padded pixels.
            eps (float, optional): Defaults to the value controlled by the constants class.
            demod (bool, optional): Whether to apply demodulation or not. Defaults to True.

        Returns:
            None:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.weight = EqualizedWeight(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.eps = eps if eps is not None else Constants.eps
        self.demod = demod

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute the modulated convolution.

        Args:
            x (torch.Tensor): 4D convolution input.
            s (torch.Tensor): 2D scaling factor.

        Returns:
            torch.Tensor: Modulated convolution result.
        """
        B, C, H, W = x.size()
        weights = self.weight().unsqueeze(0)
        weights = weights * s.reshape(B, 1, -1, 1, 1)
        # sigma is 1 / sqrt(sum(w^2))
        if self.demod:
            sigma_inv = torch.rsqrt(
                weights.square().sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )
            weights = weights * sigma_inv
        _, _, *ws = weights.shape
        weights = weights.reshape(B * self.out_channels, *ws)
        x = x.reshape(1, -1, H, W)
        # unbiased convolution
        x = F.conv2d(x, weights, padding=self.padding, groups=B)
        return x.reshape(-1, self.out_channels, H, W)


class Blur(nn.Module):
    """A blurring layer."""

    def __init__(self) -> None:
        """Intiialize the blurring layer."""
        super().__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float)[
            None, None, ...
        ]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel)
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a blurring pass.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        b, c, h, w = x.size()
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)

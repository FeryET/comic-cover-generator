"""Base Module."""
import math

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


class EqualLinear(nn.Linear):
    """An Equalized Linear Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Intiailize an eqaulized linear.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            bias (bool, optional): Defaults to True.
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.register_parameter(
            "scale_w",
            nn.Parameter(
                torch.as_tensor([math.sqrt(2 / in_features)]), requires_grad=True
            ),
        )
        self.weight.data.normal_()
        if self.bias is not None:
            self.bias.data = torch.ones(1)

    def forward(self, input: Tensor) -> Tensor:
        """Forward call of equalized linear.

        Args:
            input (Tensor):

        Returns:
            Tensor:
        """
        return F.linear(
            input,
            self.weight * self.scale_w,
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
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
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
            padding_mode (str, optional): Defaults to "zeros".
            device (_type_, optional): Defaults to None.
            dtype (_type_, optional): Defaults to None.
        """
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self._conv.weight.data.normal_(0, 1)

        self.register_buffer(
            "scale_w",
            torch.Tensor([math.sqrt(2 / (in_channels * (kernel_size**2)))]),
        )

        if bias is not False:
            self._bias = nn.Parameter(
                data=torch.zeros(out_channels), requires_grad=True
            )
        else:
            self._bias = None

    @property
    def weight(self) -> torch.Tensor:
        """Weight property of EqualConv2d.

        Returns:
            torch.Tensor:
        """
        return self._conv.weight * self.scale_w

    @property
    def bias(self) -> torch.Tensor:
        """Bias property of EqualConv2d.

        Returns:
            torch.Tensor:
        """
        return self._bias

    def forward(self, input: Tensor) -> Tensor:
        """Apply an equalized conv2d.

        Args:
            input (Tensor):

        Returns:
            Tensor:
        """
        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self._conv.stride,
            self._conv.padding,
            self._conv.dilation,
            self._conv.groups,
        )


class ModulatedConv2D(nn.Module):
    """Modulated convolution layer."""

    def __init__(self, eps: float = None, **conv_params) -> None:
        """Initializes a modulated convolution.

        Args:
            eps (float, optional): Defaults to the value controlled by the constants class.

        Returns:
            None:
        """
        super().__init__()
        self.eps = Constants.eps if eps is None else eps
        self.eq_conv = EqualConv2d(**conv_params)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute the modulated convolution.

        Args:
            x (torch.Tensor): 4D convolution input.
            s (torch.Tensor): 2D scaling factor.

        Returns:
            torch.Tensor: Modulated convolution result.
        """
        B = x.size(0)
        w = self.eq_conv.weight.unsqueeze(0)
        w = w * s.reshape(B, 1, -1, 1, 1)
        # sigma is 1 / sqrt(sum(w^2))
        sigma = w.square().sum(dim=(2, 3, 4)).add(self.eps).rsqrt()
        x = x * s.reshape(B, -1, 1, 1)
        x = self.eq_conv(x)
        # add bias too if applicable
        if self.eq_conv.bias is not None:
            x = x * sigma.to(x.dtype).reshape(B, -1, 1, 1) + self.eq_conv.bias.reshape(
                1, -1, 1, 1
            )
        else:
            x = x * sigma.to(x.dtype).reshape(B, -1, 1, 1)
        return x

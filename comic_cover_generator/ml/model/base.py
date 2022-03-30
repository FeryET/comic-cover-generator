"""Base Module."""
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    PackedSequence,
    _packed_sequence_init,
    pack_sequence,
    unpack_sequence,
)

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
                torch.as_tensor([1.0 / math.sqrt(in_features)]), requires_grad=True
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
        self._conv.weight.data.normal_()

        self.register_buffer(
            "scale_w",
            torch.Tensor([math.sqrt(2 / (in_channels * (kernel_size**2)))]),
        )

        if bias is not False:
            self._bias = nn.Parameter(
                data=torch.zeros(out_channels), requires_grad=True
            )
            self.register_buffer(
                "scale_b",
                torch.Tensor([math.sqrt(2 / out_channels)]),
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
        if self._bias is not None:
            return self._bias * self.scale_b
        else:
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


class PackSequenceEmbedding(nn.Embedding):
    """An embedding layer which accepts packed sequences as input."""

    def forward(self, packed_ids: PackedSequence) -> PackedSequence:
        """Map a packed sequence to an embeded packed sequence.

        Args:
            packed_ids (PackedSequence): packed id sequence.

        Returns:
            PackedSequence: Embedded packed sequence.
        """
        embeds = super().forward(packed_ids.data.long())
        return _packed_sequence_init(
            embeds,
            packed_ids.batch_sizes,
            packed_ids.sorted_indices,
            packed_ids.unsorted_indices,
        ).to(embeds.device)


class Seq2Vec(nn.Module):
    """A layer which maps a sequence of varying length to vectors."""

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_characters: int = 256,
        padding_idx: int = 0,
    ) -> None:
        """Initialize a seq2vec instance.

        Args:
            embed_dim (int, optional): Defaults to 32.
            hidden_size (int, optional): Defaults to 64.
            num_layers (int, optiona): Defaults to 2.
            n_characters (int, optional): vocab size. Defaults to 256.
            padding_idx (int, optional): Defaults to 0.
        """
        super().__init__()
        self.embed = PackSequenceEmbedding(
            n_characters, embed_dim, padding_idx=padding_idx
        )

        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bias=True,
            bidirectional=True,
        )

    def forward(self, seq: Sequence[Tensor]) -> Tensor:
        """Map sequence to vector.

        Args:
            seq (Sequence[Tensor]):

        Returns:
            Tensor:
        """
        # pack the sequences
        seq_packed = pack_sequence(
            seq,
            enforce_sorted=True,
        ).to(seq[0].device)

        # embed the packed sequence
        x = self.embed(seq_packed)

        # get the lstm output
        x, _ = self.gru(x)

        # keep only the last cell's output
        x = torch.stack([row[0] for row in unpack_sequence(x)])

        return x


class ModulatedConv2D(nn.Module):
    """Modulated convolution layer."""

    def __init__(self, eps: float = 1e-8, **conv_params) -> None:
        """Initializes a modulated convolution.

        Args:
            eps (float, optional): Defaults to 1e-8.

        Returns:
            None:
        """
        super().__init__()
        self.eps = eps
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

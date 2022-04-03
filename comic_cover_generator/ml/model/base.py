"""Base Module."""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, _packed_sequence_init
from transformers import AutoConfig, AutoModel, BatchEncoding, BertConfig

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.typing import Protocol, TypedDict


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


class PositionalEncoding(nn.Module):
    """Positional Encoding Layer similar to Attention is All You Need."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initializes a positional encoding layer.

        Args:
            d_model (int): embedding dimension.
            dropout (float, optional): Defaults to 0.1.
            max_len (int, optional):  Defaults to 100.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Apply positional encoding on input.

        Args:
            x: Tensor, shape [batch_size, seq, embedding_dim]
        """
        x = x + self.pe[:, x.size(1)]
        return self.dropout(x)


class Seq2VecParams(TypedDict):
    """Parameters of Seq2Vec model."""

    transformer_model: str


class Seq2Vec(nn.Module, Freezeable):
    """A layer which maps a sequence of varying length to vectors."""

    def __init__(self, transformer_model: str) -> None:
        """Initialize a seq2vec instance.

        Args:
            transformer_model (str):
            max_length (int):

        Returns:
            None:
        """
        super().__init__()

        self.transformer = AutoModel.from_pretrained(
            transformer_model, cache_dir=Constants.cache_dir
        )
        self.transformer.train()
        self.transformer_config: BertConfig = AutoConfig.from_pretrained(
            transformer_model, cache_dir=Constants.cache_dir
        )

        self.hidden_size = self.transformer_config.hidden_size

        if self.transformer_config.model_type == "bert":
            self.transformer.pooler.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            self.pooler = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size, bias=True),
                nn.LeakyReLU(negative_slope=0.1),
            )

    def forward(self, seq: BatchEncoding) -> Tensor:
        """Map sequence to vector.

        Args:
            seq (BatchEncoding):

        Returns:
            Tensor:
        """
        x = self.transformer(**seq)
        if "pooler_output" in x:
            x = x.pooler_output
        else:
            x = self.pooler(x.last_hidden_state[:, 0])
        return x

    def freeze(self):
        """Freeze generator layers."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze generator layers.

        Raises:
            TypeError:
        """
        self.freeze()
        if self.transformer_config.model_type == "bert":
            for p in self.transformer.pooler.parameters():
                p.requires_grad = True
        else:
            raise TypeError("Currently only bert models are supported.")


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

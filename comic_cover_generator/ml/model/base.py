"""Base Module."""
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import (
    PackedSequence,
    _packed_sequence_init,
    pack_sequence,
    unpack_sequence,
)

from comic_cover_generator.typing import Protocol


def equal_lr(module: nn.Module) -> nn.Module:
    """Apply equalizing learning rate to a module.

    Args:
        module (nn.Module):

    Returns:
        nn.Module:
    """
    EqualLR.apply(module, "weight")
    return module


def EqualLinear(*args, **kwargs) -> nn.Linear:
    """Instantiate an equalized lr linear layer.

    Returns:
        nn.Linear:
    """
    linear = nn.Linear(*args, **kwargs)
    linear.weight.data.normal_()
    if linear.bias is not None:
        linear.bias.data.zero_()
    return equal_lr(linear)


def EqualConv2d(*args, **kwargs) -> nn.Conv2d:
    """Instantiate an equalized lr Conv2d layer.

    Returns:
        nn.Conv2d:
    """
    conv = nn.Conv2d(*args, **kwargs)
    conv.weight.data.normal_()
    if conv.bias is not None:
        conv.bias.data.zero_()
    return equal_lr(conv)


class EqualLR:
    """Equalize learning rate for a layer.

    Courtesy of: https://github.com/rosinality/style-based-gan-pytorch/blob/07fa60be77b093dd13a46597138df409ffc3b9bc/model.py#L380
    """

    def __init__(self, name: str):
        """Initialize an equalized learning rate transform for a layer.

        Args:
            name (str): name of the equalized parameter.
        """
        self.name = name

    def compute_weight(self, module: nn.Module) -> torch.Tensor:
        """Compute the equalized parameter.

        Args:
            module (nn.Module):

        Returns:
            torch.Tensor:
        """
        weight = getattr(module, self.name + "_orig")
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module: nn.Module, name: str) -> None:
        """Apply the equalized learning rate transform on a module.

        Args:
            module (nn.Module):
            name (str):

        Returns:
            None:
        """
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + "_orig", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

    def __call__(self, module: nn.Module, *args) -> None:
        """Update the module with equalized weights.

        Args:
            module (nn.Module):

        Returns:
            None:
        """
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class Freezeable(Protocol):
    """Protocol for freezable nn.Modules."""

    def freeze(self):
        """Freezing method."""
        ...

    def unfreeze(self):
        """Unfreezing method."""
        ...


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

    def __init__(self, eps: float = 1e-5, **conv_params) -> None:
        """Initializes a modulated convolution.

        Args:
            eps (float, optional): Defaults to 1e-5.

        Returns:
            None:
        """
        super().__init__()
        self.eps = eps
        self.conv = EqualConv2d(**conv_params)
        if self.conv.bias is not None:
            self.bias = torch.nn.Parameter(
                self.conv.bias.data.reshape(1, -1, 1, 1), requires_grad=True
            )
            self.conv.bias = None

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute the modulated convolution.

        Args:
            x (torch.Tensor): 4D convolution input.
            s (torch.Tensor): 2D scaling factor.

        Returns:
            torch.Tensor: Modulated convolution result.
        """
        B = x.size(0)
        w = self.conv.weight_orig.unsqueeze(0)
        w = w * s.reshape(B, 1, -1, 1, 1)
        # sigma is 1 / sqrt(sum(w^2))
        sigma = w.square().sum(dim=(2, 3, 4)).add(self.eps).rsqrt()
        x = x * s.reshape(B, -1, 1, 1)
        x = self.conv(x)
        # add bias too if applicable
        x = x * sigma.to(x.dtype).reshape(B, -1, 1, 1) + self.bias
        return x

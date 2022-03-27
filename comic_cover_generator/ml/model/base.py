"""Base Module."""
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

    embed_dim = 16

    def __init__(self, hidden_size=512, n_characters=256, padding_idx=0) -> None:
        """Initialize a seq2vec instance.

        Args:
            hidden_size (int, optional): Defaults to 512.
            n_characters (int, optional): vocab size. Defaults to 256.
            padding_idx (int, optional): Defaults to 0.
        """
        super().__init__()
        self.embed = PackSequenceEmbedding(
            n_characters, self.embed_dim, padding_idx=padding_idx
        )

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
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
        x, _ = self.lstm(x)

        # keep only the last cell's output
        x = torch.stack([row[0] for row in unpack_sequence(x)])

        # average the directions
        x = (x[..., : self.hidden_size] + x[..., self.hidden_size :]) / 2

        return x


class ModulatedConv2D(nn.Module):
    """Modulated convolution layer."""

    def __init__(self, eps: float = 1e-5, **conv_params) -> None:
        """Initializes a modulated convolution.

        Args:
            eps (float, optional): _description_. Defaults to 1e-5.

        Returns:
            None:
        """
        super().__init__()
        self.eps = eps
        self.conv = nn.Conv2d(**conv_params)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Compute the modulated convolution.

        Args:
            x (torch.Tensor): 4D convolution input.
            s (torch.Tensor): 2D scaling factor.

        Returns:
            torch.Tensor: Modulated convolution result.
        """
        B = x.size(0)
        w = self.conv.weight.unsqueeze(0)
        w = w * s.reshape(B, 1, -1, 1, 1)
        # sigma is 1 / sqrt(sum(w^2))
        sigma = w.square().sum(dim=(2, 3, 4)).add(self.eps).rsqrt()
        x = x * s.reshape(B, -1, 1, 1)
        x = F.conv2d(
            input=x,
            weight=self.conv.weight.to(x.dtype),
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )

        # add bias too if applicable
        x = x * sigma.to(x.dtype).reshape(B, -1, 1, 1) + self.conv.bias.reshape(
            1, -1, 1, 1
        )
        return x

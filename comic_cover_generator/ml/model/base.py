"""Base Module."""
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
        )


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

        self.vectorizer = nn.LSTM(
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
        seq_packed = pack_sequence(
            seq,
            enforce_sorted=False,
        )

        x = self.embed(seq_packed)

        x, _ = self.vectorizer(x)

        x = torch.stack([row[0] for row in unpack_sequence(x)])

        x = (x[..., : self.hidden_size] + x[..., self.hidden_size :]) / 2

        return x

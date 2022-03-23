"""Generator module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import (
    Freezeable,
    ResNetBlock,
    ResNetScaler,
    Seq2Vec,
)


class Generator(nn.Module, Freezeable):
    """Generator model based on PGAN."""

    latent_dim: int = 512
    output_shape: Tuple[int, int] = (128, 128)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()

        self.condition = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(128, 2, 2)),
            ResNetBlock(128, 0.2),
            ResNetBlock(128, 0.2),
            ResNetBlock(128, 0.2),
        )

        self.title_embed = nn.Sequential(
            Seq2Vec(64),
            nn.Unflatten(dim=-1, unflattened_size=(16, 2, 2)),
            nn.Conv2d(16, 128, kernel_size=1, padding=0, stride=1, bias=0, groups=16),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )

        self.features = nn.Sequential(
            ResNetScaler("up", 128, 256, kernel_size=4, stride=4, padding=0),
            ResNetBlock(256, 0.2),
            ResNetBlock(256, 0.2),
            ResNetBlock(256, 0.2),
            ResNetBlock(256, 0.2),
            ResNetBlock(256, 0.2),
            ResNetBlock(256, 0.2),
            ResNetScaler("up", 256, 512, kernel_size=4, stride=4, padding=0),
            ResNetBlock(512, 0.2),
            ResNetBlock(512, 0.2),
            ResNetBlock(512, 0.2),
            ResNetBlock(512, 0.2),
            ResNetBlock(512, 0.2),
            ResNetBlock(512, 0.2),
            ResNetScaler("up", 512, 128, kernel_size=4, stride=4, padding=0),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, z: torch.Tensor, title_seq: torch.Tensor) -> torch.Tensor:
        """Map a noise and a sequence to an image.

        Args:
            z (torch.Tensor): Noise input.
            title_seq (torch.Tensor): title sequence.

        Returns:
            torch.Tensor:
        """
        x = self.condition(z) + self.title_embed(title_seq)
        return self.to_rgb(self.features(x))

    def freeze(self):
        """Freeze the generator model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the generator model."""
        for p in self.parameters():
            p.requires_grad = True
        return self

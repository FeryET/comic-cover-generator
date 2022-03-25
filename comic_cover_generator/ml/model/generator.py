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
            nn.Unflatten(dim=-1, unflattened_size=(8, 8, 8)),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(1, 16, affine=True),
            nn.LeakyReLU(0.1),
        )

        self.title_embed = nn.Sequential(
            Seq2Vec(64),
            nn.Unflatten(dim=-1, unflattened_size=(1, 8, 8)),
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(1, 16, affine=True),
            nn.LeakyReLU(0.1),
        )

        self.features = nn.Sequential(
            ResNetScaler("up", 16, 32, kernel_size=2, stride=2, padding=0),
            # 16x16
            nn.Sequential(*[ResNetBlock(32, expansion=2) for _ in range(3)]),
            ResNetScaler("up", 32, 64, kernel_size=2, stride=2, padding=0),
            # 32x32
            nn.Sequential(*[ResNetBlock(64, expansion=2) for _ in range(3)]),
            ResNetScaler("up", 64, 128, kernel_size=2, stride=2, padding=0),
            # 64x64
            nn.Sequential(*[ResNetBlock(128, expansion=2) for _ in range(3)]),
            ResNetScaler("up", 128, 256, kernel_size=2, stride=2, padding=0),
            # 128x128
            nn.Sequential(*[ResNetBlock(256, expansion=2) for _ in range(3)]),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0), nn.Tanh()
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

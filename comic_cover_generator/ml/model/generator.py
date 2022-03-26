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
    output_shape: Tuple[int, int] = (64, 64)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()

        self.condition = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(32, 4, 4)),
            nn.Conv2d(32, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.title_embed = nn.Sequential(
            Seq2Vec(64),
            nn.Unflatten(dim=-1, unflattened_size=(4, 4, 4)),
            nn.Conv2d(4, 1024, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

        self.features = nn.Sequential(
            # 4x4
            ResNetScaler(
                "up",
                1024,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # 8x8
            nn.Sequential(*[ResNetBlock("gen", 512, expansion=1) for _ in range(2)]),
            ResNetScaler("up", 512, 256, kernel_size=4, stride=2, padding=1),
            # 16x16
            nn.Sequential(*[ResNetBlock("gen", 256, expansion=1) for _ in range(2)]),
            ResNetScaler("up", 256, 128, kernel_size=4, stride=2, padding=1),
            # 32x32
            nn.Sequential(*[ResNetBlock("gen", 128, expansion=2) for _ in range(3)]),
        )
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
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

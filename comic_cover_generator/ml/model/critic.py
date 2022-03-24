"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import Freezeable, ResNetBlock, ResNetScaler


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (128, 128)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        self.features = nn.Sequential(
            ResNetScaler("down", 3, 64, 7, stride=4, padding=3),
            ResNetBlock(64),
            ResNetBlock(64),
            ResNetBlock(64),
            ResNetScaler("down", 64, 128, 5, stride=4, padding=1),
            ResNetBlock(128, 2),
            ResNetBlock(128, 2),
            ResNetBlock(128, 2),
            ResNetScaler("down", 128, 256, 3, stride=2, padding=1),
            ResNetBlock(256, 2),
            ResNetBlock(256, 2),
            ResNetBlock(256, 2),
            ResNetScaler("down", 256, 512, 3, stride=2, padding=1),
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(), nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.clf(self.features(x))

    def freeze(self):
        """Freeze the critic model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the critic model."""
        for p in self.parameters():
            p.requires_grad = True
        return self

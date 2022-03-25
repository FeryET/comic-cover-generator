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
            ResNetScaler("down", 3, 24, 7, stride=4, padding=2),
            # 32 x 32
            nn.Sequential(*[ResNetBlock("critic", 24, expansion=1) for _ in range(2)]),
            ResNetScaler("down", 24, 96, 5, stride=4, padding=1),
            # 8 x 8
            nn.Sequential(*[ResNetBlock("critic", 96, expansion=1) for _ in range(2)]),
            ResNetScaler("down", 96, 192, 3, stride=2, padding=1),
            # 4 x 4
            nn.Sequential(*[ResNetBlock("critic", 192, expansion=1) for _ in range(2)]),
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(), nn.Linear(192, 1)
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

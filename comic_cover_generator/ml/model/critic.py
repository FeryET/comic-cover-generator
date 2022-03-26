"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import Freezeable, ResNetBlock, ResNetScaler
from comic_cover_generator.ml.model.utils import weights_init


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (64, 64)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        self.features = nn.Sequential(
            # 64 x 64
            ResNetScaler("down", 3, 128, 7, stride=4, padding=3),
            # 16 x 16
            nn.Sequential(*[ResNetBlock("critic", 128, expansion=1) for _ in range(2)]),
            ResNetScaler("down", 128, 256, 3, stride=2, padding=1),
            # 8 x 8
            nn.Sequential(*[ResNetBlock("critic", 256, expansion=1) for _ in range(3)]),
            ResNetScaler("down", 256, 512, 3, stride=2, padding=1),
            # 4 x 4
            nn.Sequential(*[ResNetBlock("critic", 512, expansion=2) for _ in range(4)]),
        )

        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 1),
        )

        self.apply(weights_init)

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

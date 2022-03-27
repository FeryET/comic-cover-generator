"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import Freezeable
from comic_cover_generator.ml.model.utils import weights_init


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (64, 64)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 512, 1))

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

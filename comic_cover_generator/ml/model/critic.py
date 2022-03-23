"""Critic Module."""
from typing import Tuple

import torch
from torch import nn

from comic_cover_generator.ml.model.base import Freezeable, ResNetBlock


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (128, 128)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=4, padding=0, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            ResNetBlock(64, p_dropout=0.2),
            ResNetBlock(64, p_dropout=0.2),
            ResNetBlock(64, p_dropout=0.2),
            nn.Conv2d(64, 128, 5, stride=4, padding=0, bias=False),
            ResNetBlock(128, p_dropout=0.2),
            ResNetBlock(128, p_dropout=0.2),
            ResNetBlock(128, p_dropout=0.2),
            nn.Conv2d(128, 256, 3, stride=4, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.clf = nn.Linear(256, 1)

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

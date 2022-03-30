"""Base module for training strategy subpackage."""
from typing import List

import torch

from comic_cover_generator.typing import Protocol, TypedDict


class TrainingStrategy(Protocol):
    """Training strategy protocol."""

    class CRITIC_LOOP_RESULT(TypedDict):
        """Result of TrainingStrategy.critic_loop."""

        loss: torch.Tensor

    class GENERATOR_LOOP_RESULT(TypedDict):
        """Result of TrainingStrategy.generator_loop."""

        loss: torch.Tensor
        fakes: torch.Tensor

    def generator_loop(
        self, seq: List[torch.Tensor], z: torch.Tensor
    ) -> GENERATOR_LOOP_RESULT:
        """Attempt a generator loop.

        Args:
            seq (List[torch.Tensor]): Input sequence.
            z (torch.Tensor): Latent noise.

        Returns:
            GENERATOR_LOOP_RESULT:
        """
        ...

    def critic_loop(
        self, reals: torch.Tensor, seq: List[torch.Tensor], z: torch.Tensor
    ) -> CRITIC_LOOP_RESULT:
        """Attempt a critic loop.

        Args:
            reals (torch.Tensor): Real images.
            seq (List[torch.Tensor]): Input sequences.
            z (torch.Tensor): Latent noise.

        Returns:
            CRITIC_LOOP_RESULT:
        """
        ...

    def attach_model(self, model: torch.nn.Module) -> None:
        """Attach a model to training strategy.

        Args:
            model (torch.nn.Module):

        Returns:
            None:
        """
        ...

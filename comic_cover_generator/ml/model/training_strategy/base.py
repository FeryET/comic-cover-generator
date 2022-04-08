"""Base module for training strategy subpackage."""

import torch
from transformers import BatchEncoding

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

    class VALIDATION_LOOP_RESULT(TypedDict):
        """Result of TrainingStrategy.validation_loop."""

        generator_loss: torch.Tensor
        critic_loss: torch.Tensor
        fakes: torch.Tensor

    def generator_loop(
        self,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> GENERATOR_LOOP_RESULT:
        """Attempt a generator loop.

        Args:
            seq (BatchEncoding):: Input sequence.
            z (torch.Tensor): Latent noise.
            batch_idx (int): Index of the input batch.
            stochastic_noise (torch.Tensor): Stochastic noise.
            optimizer_idx (int): Index of the optimizer.

        Returns:
            GENERATOR_LOOP_RESULT:
        """
        ...

    def critic_loop(
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> CRITIC_LOOP_RESULT:
        """Attempt a critic loop.

        Args:
            reals (torch.Tensor): Real images.
            seq (BatchEncoding):: Input sequences.
            z (torch.Tensor): Latent noise.
            stochastic_noise (torch.Tensor): Stochastic noise.
            batch_idx (int): Index of the input batch.
            optimizer_idx (int): Index of the optimizer.

        Returns:
            CRITIC_LOOP_RESULT:
        """
        ...

    def validation_loop(
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
    ) -> VALIDATION_LOOP_RESULT:
        """Apply a validation loop.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding)::
            z (torch.Tensor):
            batch_idx (int):

        Returns:
            VALIDATION_LOOP_RESULT:
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

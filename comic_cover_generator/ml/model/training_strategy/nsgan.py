"""Loss module."""
from typing import List

import torch
import torch.nn.functional as F

from comic_cover_generator.ml.model.diffaugment import diff_augment
from comic_cover_generator.ml.model.gan import GAN
from comic_cover_generator.ml.model.training_strategy.base import TrainingStrategy


@torch.jit.script
def generator_loss(logits_g: torch.Tensor) -> torch.Tensor:
    """Get generator loss for NSGAN.

    Args:
        logits_g (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    loss = F.softplus(-logits_g).mean()
    return loss


@torch.jit.script
def critic_loss(
    logits_r: torch.Tensor,
    logits_f: torch.Tensor,
    generated_weight: float = 1,
    real_weight: float = 1,
) -> torch.Tensor:
    """Get critic loss for negative NSGAN.

    Args:
        logits_r (torch.Tensor):
        logits_f (torch.Tensor):
        generated_weight (float, optional): Defaults to 1.
        real_weight (float, optional): Defaults to 1.

    Returns:
        torch.Tensor:
    """
    lr = F.softplus(-logits_r).mean() * real_weight
    lf = F.softplus(logits_f).mean() * generated_weight
    loss = lr + lf
    return loss


class NSGANTrainingStrategy:
    """Non Saturated GAN training strategy."""

    def __init__(
        self,
        regularization_penalty: float,
        regularization_intervel: int,
    ) -> None:
        """Initialize an NSGAN training strategy.

        Args:
            regularization_penalty (float):
            regularization_intervel (int):

        Returns:
            None:
        """
        self.regularization_penalty = regularization_penalty
        self.regularization_interval = regularization_intervel
        self.model: GAN = None

    def attach_model(self, model: GAN) -> None:
        """Attach a model to this strategy.

        Args:
            model (GAN):

        Returns:
            None:
        """
        self.model = model

    def critic_loop(
        self, reals: torch.Tensor, seq: List[torch.Tensor], z: torch.Tensor
    ) -> TrainingStrategy.CRITIC_LOOP_RESULT:
        """Apply a critic training loop.

        Args:
            reals (torch.Tensor):
            seq (List[torch.Tensor]):
            z (torch.Tensor):

        Returns:
            TrainingStrategy.CRITIC_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq)

        fakes = diff_augment(fakes, self.model.augmentation_policy)
        reals = diff_augment(reals, self.model.augmentation_policy)

        logits_r = self.model.critic(reals)
        logits_f = self.model.critic(fakes)

        loss = critic_loss(logits_r, logits_f)

        return {"loss": loss}

    def generator_loop(
        self, seq: List[torch.Tensor], z: torch.Tensor
    ) -> TrainingStrategy.GENERATOR_LOOP_RESULT:
        """Apply a generator loop.

        Args:
            seq (List[torch.Tensor]):
            z (torch.Tensor):

        Returns:
            TrainingStrategy.GENERATOR_LOOP_RESULT:
        """
        self.model.critic.freeze()
        self.model.critic.eval()

        self.model.generator.unfreeze()
        self.model.generator.train()

        fakes = self.model.generator(z, seq)
        logits_f = self.model.critic(fakes)

        loss = generator_loss(logits_f)

        return {"loss": loss}

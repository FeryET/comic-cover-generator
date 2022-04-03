"""Loss module."""

import torch
from transformers import BatchEncoding

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.model.diffaugment import diff_augment
from comic_cover_generator.ml.model.gan import GAN
from comic_cover_generator.ml.model.training_strategy.base import TrainingStrategy


@torch.jit.script
def generator_loss(logits_f: torch.Tensor, eps: float) -> torch.Tensor:
    """Get generator loss for NSGAN.

    Args:
        logits_f (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    loss = -torch.mean(torch.log(torch.sigmoid(logits_f) + eps))
    return loss


def critic_loss(
    logits_r: torch.Tensor,
    logits_f: torch.Tensor,
    eps: float,
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
    loss = -torch.mean(
        torch.log(torch.sigmoid(logits_r) + eps) * real_weight
        + torch.log(1 - torch.sigmoid(logits_f) + eps) * generated_weight
    )
    return loss


class NSGANTrainingStrategy:
    """Non Saturated GAN training strategy."""

    def __init__(
        self,
        regularization_penalty: float,
        regularization_interval: int,
    ) -> None:
        """Initialize an NSGAN training strategy.

        Args:
            regularization_penalty (float):
            regularization_interval (int):

        Returns:
            None:
        """
        self.regularization_penalty = regularization_penalty
        self.regularization_interval = regularization_interval
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
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        batch_idx: int,
    ) -> TrainingStrategy.CRITIC_LOOP_RESULT:
        """Apply a critic training loop.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding)::
            z (torch.Tensor):

        Returns:
            TrainingStrategy.CRITIC_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq)

        reals.requires_grad_(True)

        fakes = diff_augment(fakes, self.model.augmentation_policy)
        reals = diff_augment(reals, self.model.augmentation_policy)

        logits_r = self.model.critic(reals)
        logits_f = self.model.critic(fakes)

        if batch_idx % self.regularization_interval == 0:
            (grad_real,) = torch.autograd.grad(
                outputs=logits_f.sum(),
                inputs=reals,
                create_graph=True,
                retain_graph=True,
            )
            grad_penalty = (
                grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            )
        else:
            grad_penalty = 0
        loss = (
            critic_loss(logits_r, logits_f, eps=Constants.eps)
            + grad_penalty * self.regularization_penalty
        )

        return {"loss": loss}

    def generator_loop(
        self, seq: BatchEncoding, z: torch.Tensor, batch_idx: int
    ) -> TrainingStrategy.GENERATOR_LOOP_RESULT:
        """Apply a generator loop.

        Args:
            seq (BatchEncoding)::
            z (torch.Tensor):

        Returns:
            TrainingStrategy.GENERATOR_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq)
        logits_f = self.model.critic(fakes)

        loss = generator_loss(logits_f, eps=Constants.eps)

        return {"loss": loss, "fakes": fakes}

    def validation_loop(
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        batch_idx: int,
    ) -> TrainingStrategy.VALIDATION_LOOP_RESULT:
        """Apply a validation loop.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding)::
            z (torch.Tensor):

        Returns:
            TrainingStrategy.VALIDATION_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq)

        logits_r = self.model.critic(reals)
        logits_f = self.model.critic(fakes)

        return {
            "generator_loss": generator_loss(logits_f, eps=Constants.eps),
            "critic_loss": critic_loss(logits_r, logits_f, eps=Constants.eps),
            "fakes": fakes,
        }

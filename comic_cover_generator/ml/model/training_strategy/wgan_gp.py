"""Loss module."""

import torch
from torch import nn
from transformers import BatchEncoding

from comic_cover_generator.ml.model.diffaugment import diff_augment
from comic_cover_generator.ml.model.gan import GAN
from comic_cover_generator.ml.model.training_strategy.base import TrainingStrategy


def generator_loss_fn(critic_generated_images_prediction: torch.Tensor) -> torch.Tensor:
    """Compute generator loss.

    Args:
        critic_generated_images_prediction (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return -1.0 * torch.mean(critic_generated_images_prediction)


def critic_loss_fn(
    critic_generated_images_prediction: torch.Tensor,
    critic_real_images_prediction: torch.Tensor,
    gradient_penalty: torch.Tensor,
    gradient_pentaly_coef: int,
) -> torch.Tensor:
    """Compute critic loss.

    Args:
        critic_generated_images_prediction (torch.Tensor): Critic prediction on fake images.
        critic_real_images_prediction (torch.Tensor): Critic prediction on real images.
        gradient_penalty (torch.Tensor): Gradient penalty.
        gradient_pentaly_coef (int): Gradient penalty coefficient.

    Returns:
        torch.Tensor:
    """
    return (
        torch.mean(critic_generated_images_prediction)
        - torch.mean(critic_real_images_prediction)
        + gradient_penalty * gradient_pentaly_coef
    )


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """Compute gradient penalty.

    Args:
        critic (nn.Module):
        real (torch.Tensor):
        fake (torch.Tensor):


    Returns:
        torch.Tensor:
    """
    alpha = torch.rand(real.size(0)).to(real.device)

    alpha = alpha.expand(*real.size()[::-1])
    alpha = alpha.permute(*torch.arange(real.ndim - 1, -1, -1))
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images = torch.autograd.Variable(
        interpolated_images, requires_grad=True
    )

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = torch.linalg.norm(gradient, 2, dim=-1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


class WGANPlusGPTrainingStrategy:
    """WGAN with Gradient Penalty TrainingStrategy."""

    def __init__(self, gradient_penalty_coef: int, critic_update_interval: int) -> None:
        """Initialize a WGAN-GP training strategy.

        Args:
            gradient_penalty_coef (int): Gradient penalty coefficient.
            critic_update_interval (int): Critic repeated update intervals.
        """
        self.gradient_penalty_coef = gradient_penalty_coef
        self.critic_update_interval = critic_update_interval
        self.model: GAN = None

    def attach_model(self, model: GAN) -> None:
        """Attach a model to training strategy.

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
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> TrainingStrategy.CRITIC_LOOP_RESULT:
        """Apply a critic training loop for WGAN-GP.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding):
            z (torch.Tensor):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            TrainingStrategy.CRITIC_LOOP_RESULT:
        """
        fakes = self.model.generator._train_step_forward(z, seq, stochastic_noise)

        fakes = diff_augment(fakes, self.model.augmentation_policy)
        reals = diff_augment(reals, self.model.augmentation_policy)

        critic_score_fakes = self.model.critic(fakes)
        critic_score_reals = self.model.critic(reals)
        gp = gradient_penalty(self.model.critic, reals.data, fakes.data)
        loss_critic = critic_loss_fn(
            critic_score_fakes, critic_score_reals, gp, self.gradient_penalty_coef
        )
        return {"loss": loss_critic}

    def generator_loop(
        self,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> TrainingStrategy.GENERATOR_LOOP_RESULT:
        """Apply a generator training loop for WGAN-GP.

        Args:
            seq (BatchEncoding):
            z (torch.Tensor):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            TrainingStrategy.GENERATOR_LOOP_RESULT:
        """
        fakes = self.model.generator._train_step_forward(z, seq, stochastic_noise)
        fakes = diff_augment(fakes, self.model.augmentation_policy)

        critic_gen_fake = self.model.critic(fakes).reshape(-1)
        loss_gen = generator_loss_fn(critic_gen_fake)

        return {"loss": loss_gen, "fakes": fakes}

    def validation_loop(
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
    ) -> TrainingStrategy.VALIDATION_LOOP_RESULT:
        """Apply a validation loop for WGAN-GP.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding)::
            z (torch.Tensor):
            batch_idx (int):

        Returns:
            TrainingStrategy.VALIDATION_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq, stochastic_noise)
        critic_score_fakes = self.model.critic(fakes)
        critic_score_reals = self.model.critic(reals)
        loss_critic = critic_loss_fn(critic_score_fakes, critic_score_reals, 0, 0)

        loss_generator = generator_loss_fn(critic_score_fakes)

        return {
            "generator_loss": loss_generator,
            "critic_loss": loss_critic,
            "fakes": fakes,
        }

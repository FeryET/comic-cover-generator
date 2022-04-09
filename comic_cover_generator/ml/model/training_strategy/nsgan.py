"""Loss module."""

import math
from typing import Tuple

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
    return -torch.log(torch.sigmoid(logits_f) + eps).mean()


@torch.jit.script
def critic_loss(
    logits_r: torch.Tensor,
    logits_f: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Get critic loss for negative NSGAN.

    Args:
        logits_r (torch.Tensor):
        logits_f (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    f_loss = -torch.log(1 - torch.sigmoid(logits_f) + eps)
    r_loss = -torch.log(torch.sigmoid(logits_r) + eps)
    return r_loss.mean() + f_loss.mean()


def compute_r1_regularization(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Compute r1 gradient penalty.

    Args:
        outputs (torch.Tensor):
        inputs (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    (grad_real,) = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=inputs,
        create_graph=True,
    )
    grad_penalty = grad_real.pow(2).flatten(1).sum(1).mean()
    return grad_penalty


def compute_path_length_regularization(
    fakes: torch.Tensor,
    w: torch.Tensor,
    mean_path_length: torch.Tensor,
    beta: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute path length regualization.

    Args:
        fakes (torch.Tensor): Fake images 4D.
        w (torch.Tensor): Mapping network input for generator blocks. 3D.
        mean_path_length (torch.Tensor):  previous mean average path length. scaler.
        beta (float, optional): Decay coefficient. Defaults to 0.99.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    noise = torch.randn_like(fakes) / math.sqrt(fakes.size(2) * fakes.size(3))
    (grad,) = torch.autograd.grad(
        outputs=(fakes * noise).sum(), inputs=w, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + (1 - beta) * (path_lengths.mean() - mean_path_length)
    pl_penalty = (path_lengths - path_mean).pow(2).mean()
    return pl_penalty, path_mean.detach(), path_lengths


def _interval_condition(batch_idx, interval):
    return (batch_idx // 2) % interval == interval - 1


class NSGANTrainingStrategy:
    """Non Saturated GAN training strategy."""

    def __init__(
        self,
        r1_penalty_coef: float,
        pl_penalty_coef: float,
        r1_regularization_interval: int,
        pl_regularization_interval: int,
        pl_start_from_iteration: int,
        path_length_beta: float = 0.99,
        eps: float = None,
    ) -> None:
        """Initialize an NSGAN training strategy.

        Args:
            r1_penalty (float):
            r1_regularization_interval (int):
            pl_regularization_interval (int):
            pl_start_from_iteration (int):
            path_length_beta (float): Defaults to 0.99.
            eps (float, optinal): If none, defaults to Constants.eps.

        Returns:
            None:
        """
        self.r1_penalty_coef = r1_penalty_coef
        self.pl_penalty_coef = pl_penalty_coef
        self.r1_regularization_interval = r1_regularization_interval
        self.pl_regularization_interval = pl_regularization_interval

        # since each batch goes only to one of either of generator or discriminator
        # it's imperative to make sure that we apply these relatively
        self.scaled_r1_regualirzation_interval = r1_regularization_interval
        self.scaled_pl_regularization_interval = pl_regularization_interval
        self.pl_start_from_iteration = pl_start_from_iteration

        self.path_length_beta = path_length_beta
        self.model: GAN = None

        if eps is None:
            eps = Constants.eps
        self.eps = eps

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
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> TrainingStrategy.CRITIC_LOOP_RESULT:
        """Apply a critic training loop.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding):
            z (torch.Tensor):
            stochastic_noise (torch.Tensor):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            TrainingStrategy.CRITIC_LOOP_RESULT:
        """
        if _interval_condition(batch_idx, self.r1_regularization_interval):
            reals.requires_grad_(True)
        reals = diff_augment(reals, self.model.augmentation_policy)

        # compute the normal gan loss
        fakes = self.model.generator._train_step_forward(z, seq, stochastic_noise)
        fakes = diff_augment(fakes, self.model.augmentation_policy)

        logits_f = self.model.critic(fakes)
        logits_r = self.model.critic(reals)
        loss = critic_loss(logits_r, logits_f, self.eps)
        if _interval_condition(batch_idx, self.r1_regularization_interval):
            # compute the regularization term only
            loss = loss + (
                compute_r1_regularization(logits_r, reals)
                * self.r1_penalty_coef
                * self.r1_regularization_interval
                * 0.5
            )

        return {"loss": loss}

    def generator_loop(
        self,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int,
    ) -> TrainingStrategy.GENERATOR_LOOP_RESULT:
        """Apply a generator loop.

        Args:
            seq (BatchEncoding)::
            z (torch.Tensor):
            stochastic_noise (torch.Tensor):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            TrainingStrategy.GENERATOR_LOOP_RESULT:
        """
        fakes, w = self.model.generator._train_step_forward(
            z, seq, stochastic_noise, return_w=True
        )
        fakes = diff_augment(fakes, self.model.augmentation_policy)
        logits_f = self.model.critic(fakes)
        loss = generator_loss(logits_f, self.eps)
        if (
            self.model.global_step > self.pl_start_from_iteration
            and _interval_condition(batch_idx, self.pl_regularization_interval)
        ):
            (
                path_length_penalty,
                path_mean,
                path_lengths,
            ) = compute_path_length_regularization(
                fakes, w, self.model.generator.path_length_mean, self.path_length_beta
            )
            self.model.generator.path_length_mean = path_mean
            if not path_length_penalty.isnan():
                loss = (
                    loss
                    + path_length_penalty
                    * self.pl_penalty_coef
                    * self.pl_regularization_interval
                )

        return {"loss": loss, "fakes": fakes}

    def validation_loop(
        self,
        reals: torch.Tensor,
        seq: BatchEncoding,
        z: torch.Tensor,
        stochastic_noise: torch.Tensor,
        batch_idx: int,
    ) -> TrainingStrategy.VALIDATION_LOOP_RESULT:
        """Apply a validation loop.

        Args:
            reals (torch.Tensor):
            seq (BatchEncoding):
            z (torch.Tensor):
            stochastic_noise (torch.Tensor):
            batch_idx (int):

        Returns:
            TrainingStrategy.VALIDATION_LOOP_RESULT:
        """
        fakes = self.model.generator(z, seq, stochastic_noise)

        logits_r = self.model.critic(reals)
        logits_f = self.model.critic(fakes)

        return {
            "generator_loss": generator_loss(logits_f, self.eps),
            "critic_loss": critic_loss(logits_r, logits_f, self.eps),
            "fakes": fakes,
        }

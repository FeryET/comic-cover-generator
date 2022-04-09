"""Loss module."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BatchEncoding

from comic_cover_generator.ml.model.diffaugment import diff_augment
from comic_cover_generator.ml.model.gan import GAN
from comic_cover_generator.ml.model.training_strategy.base import TrainingStrategy


@torch.jit.script
def generator_loss(logits_f: torch.Tensor) -> torch.Tensor:
    """Get generator loss for NSGAN.

    Args:
        logits_f (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return F.softplus(-logits_f).mean()


@torch.jit.script
def critic_loss(
    logits_r: torch.Tensor,
    logits_f: torch.Tensor,
) -> torch.Tensor:
    """Get critic loss for negative NSGAN.

    Args:
        logits_r (torch.Tensor):
        logits_f (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    r_loss = F.softplus(-logits_r)
    f_loss = F.softplus(logits_f)
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


class PathLengthPenalty(nn.Module):
    """Path length penalty proposed by StyleGan2."""

    def __init__(self, beta: float = 0.99) -> None:
        """Initialize a PathLengthPenalty object.

        Args:
            beta (float, optional): The exponential decay. Defaults to 0.99.
        """
        super().__init__()
        self.register_buffer("beta", torch.as_tensor(beta))
        self.steps = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.exponential_sum = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Compute and get the penalty.

        Args:
            x (Tensor): 4D generated images.
            w (Tensor): 3D mapping network input. [n_gen_blocks, batch_size, w_dim]

        Returns:
            Tensor:
        """
        n_pixels = x.size(2) * x.size(3)
        y = torch.randn_like(x)
        o = (x * y).sum() / math.sqrt(n_pixels)
        grads, *_ = torch.autograd.grad(outputs=o, inputs=[w], create_graph=True)
        norm = grads.square().sum(2).mean(1).sqrt()
        if self.steps > 0:
            a = self.exponential_sum / (1 - self.beta.pow(self.steps))
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)
        mean = norm.mean().detach()
        self.exponential_sum.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1)
        return loss


def _interval_condition(batch_idx, interval):
    return (batch_idx // 2) % interval == interval - 1


class NSGANTrainingStrategy:
    """Non Saturated GAN training strategy."""

    def __init__(
        self,
        r1_penalty: float,
        pl_penalty: float,
        r1_regularization_interval: int,
        pl_regularization_interval: int,
        pl_start_from_iteration: int,
        path_length_beta: float = 0.99,
    ) -> None:
        """Initialize an NSGAN training strategy.

        Args:
            r1_penalty (float):
            r1_regularization_interval (int):
            pl_regularization_interval (int):
            pl_start_from_iteration (int):
            path_length_beta (float): Defaults to 0.99.

        Returns:
            None:
        """
        self.r1_penalty = r1_penalty
        self.pl_penalty = pl_penalty
        self.r1_regularization_interval = r1_regularization_interval
        self.pl_regularization_interval = pl_regularization_interval

        # since each batch goes only to one of either of generator or discriminator
        # it's imperative to make sure that we apply these relatively
        self.scaled_r1_regualirzation_interval = r1_regularization_interval
        self.scaled_pl_regularization_interval = pl_regularization_interval
        self.pl_start_from_iteration = pl_start_from_iteration

        self.plp = PathLengthPenalty(path_length_beta)
        self.model: GAN = None

    def attach_model(self, model: GAN) -> None:
        """Attach a model to this strategy.

        Args:
            model (GAN):

        Returns:
            None:
        """
        self.model = model
        self.plp = self.plp.to(model.device)

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
        loss = critic_loss(logits_r, logits_f)
        if _interval_condition(batch_idx, self.r1_regularization_interval):
            # compute the regularization term only
            loss = loss + (
                compute_r1_regularization(logits_r, reals)
                * self.r1_penalty
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
        loss = generator_loss(logits_f)
        if (
            self.model.global_step > self.pl_start_from_iteration
            and _interval_condition(batch_idx, self.pl_regularization_interval)
        ):
            if self.plp.beta.device != fakes.device:
                self.plp = self.plp.to(fakes.device)
            plp = self.plp(fakes, w)
            if not plp.isnan():
                loss = loss + plp * self.pl_penalty * self.pl_regularization_interval

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
            "generator_loss": generator_loss(logits_f),
            "critic_loss": critic_loss(logits_r, logits_f),
            "fakes": fakes,
        }

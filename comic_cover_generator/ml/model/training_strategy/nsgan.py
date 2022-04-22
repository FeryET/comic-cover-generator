"""Loss module."""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from comic_cover_generator.ml.model import Critic, Generator


@torch.jit.script
def ns_gan_gen_loss(logits_f: torch.Tensor) -> torch.Tensor:
    """Get generator loss for NSGAN.

    Args:
        logits_f (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return F.softplus(logits_f).mean()


@torch.jit.script
def ns_gan_disc_loss(
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
    f_loss = F.softplus(logits_f)
    r_loss = F.softplus(-logits_r)
    return r_loss.mean() + f_loss.mean()


def compute_r1_regularization(
    reals_pred: torch.Tensor,
    reals: torch.Tensor,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
) -> torch.Tensor:
    """Compute r1 gradient penalty.

    Args:
        reals_pred (torch.Tensor):
        reals (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    with torch.autocast(enabled=False, device_type=reals_pred.device.type):
        (grad_real,) = torch.autograd.grad(
            outputs=scaler.scale(reals_pred.sum()),
            inputs=reals,
            create_graph=True,
        )
        grad_real = grad_real / scaler.get_scale()
    grad_penalty = grad_real.pow(2).flatten(1).sum(1).mean()
    return grad_penalty


def compute_path_length_regularization(
    fakes: torch.Tensor,
    w: torch.Tensor,
    mean_path_length: torch.Tensor,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
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
    with torch.autocast(enabled=False, device_type=noise.device.type):
        (grad,) = torch.autograd.grad(
            outputs=scaler.scale((fakes * noise).sum()), inputs=w, create_graph=True
        )
        grad = grad / scaler.get_scale()
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + (1 - beta) * (path_lengths.mean() - mean_path_length)
    pl_penalty = (path_lengths - path_mean).pow(2).mean()
    return pl_penalty, path_mean.detach(), path_lengths


def nsgan_generator_train_loop(
    generator: Generator,
    critic: Critic,
    g_opt: torch.optim.Optimizer,
    g_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    z: torch.Tensor,
    title_seq: torch.Tensor,
    noise: torch.Tensor,
    batch_idx: int,
    pl_beta: float,
    pl_coef: float,
    pl_interval: float,
    grad_clip_val: float,
    autocast_enabled: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """NSGAN training loop for generator.

    Args:
        generator (Generator):
        critic (Critic):
        g_opt (torch.optim.Optimizer):
        g_scaler (torch.cuda.amp.grad_scaler.GradScaler):
        z (torch.Tensor):
        title_seq (torch.Tensor):
        noise (torch.Tensor):
        batch_idx (int):
        pl_beta (float):
        pl_coef (float):
        pl_interval (float):
        grad_clip_val (float):
        autocast_enabled (bool, optional): Defaults to False.

    Returns:
        Tuple[torch.Tensor, float, float]:
    """
    with torch.autocast(enabled=autocast_enabled, device_type=z.device.type):
        fakes, w = generator._train_step_forward(z, title_seq, noise)
        logits_f = critic(fakes)
        g_loss = ns_gan_gen_loss(logits_f)
        if batch_idx % pl_interval:
            (
                pl_penalty,
                generator.path_length_mean,
                path_lengths,
            ) = compute_path_length_regularization(
                fakes, w, generator.path_length_mean, g_scaler, pl_beta
            )
            g_loss = g_loss + pl_penalty * pl_coef * pl_interval
            path_lengths = path_lengths.detach().mean().item()
        else:
            path_lengths = None
    g_scaler.scale(g_loss).backward()
    g_scaler.unscale_(g_opt)
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=grad_clip_val)
    g_scaler.step(g_opt)
    g_scaler.update()
    return fakes, g_loss.item(), path_lengths


def nsgan_critic_train_loop(
    reals: torch.Tensor,
    fakes: torch.Tensor,
    critic: Critic,
    c_opt: torch.optim.Optimizer,
    c_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    batch_idx: int,
    autocast_enabled: bool,
    grad_clip_val: float,
    r1_interval: float,
    r1_coef: float,
) -> float:
    """NSGAN critic train loop.

    Args:
        reals (torch.Tensor):
        fakes (torch.Tensor):
        critic (Critic):
        c_opt (torch.optim.Optimizer):
        c_scaler (torch.cuda.amp.grad_scaler.GradScaler):
        batch_idx (int):
        autocast_enabled (bool):
        grad_clip_val (float):
        r1_interval (float):
        r1_coef (float):

    Returns:
        float:
    """
    with torch.autocast(enabled=autocast_enabled, device_type=reals.device.type):
        logits_r = critic(reals)
        logits_f = critic(fakes.detach())
        c_loss = ns_gan_disc_loss(logits_r, logits_f)
        if batch_idx % r1_interval:
            c_reg = compute_r1_regularization(logits_r, reals, c_scaler)
            c_loss = c_loss + c_reg * r1_coef
    c_scaler.scale(c_loss).backward()
    c_scaler.unscale_(c_opt)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=grad_clip_val)
    c_scaler.step(c_opt)
    c_scaler.update()
    return c_loss.item()

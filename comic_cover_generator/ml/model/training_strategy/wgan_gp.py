"""Loss module."""

import torch
from torch import nn


def wgan_gp_generator_loss(
    fakes_pred: torch.Tensor,
) -> torch.Tensor:
    """Compute generator loss.

    Args:
        fakes_pred (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    return -1.0 * torch.mean(fakes_pred)


def wgan_gp_critic_loss(
    fakes_pred: torch.Tensor,
    reals_pred: torch.Tensor,
) -> torch.Tensor:
    """Compute critic loss.

    Args:
        fakes_pred (torch.Tensor): Critic prediction on fake images.
        reals_pred (torch.Tensor): Critic prediction on real images.

    Returns:
        torch.Tensor:
    """
    return torch.mean(fakes_pred) - torch.mean(reals_pred)


def wgan_gp_gradient_penalty(
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

"""Loss module."""
import torch
from torch import nn


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

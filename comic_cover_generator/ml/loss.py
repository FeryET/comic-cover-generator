"""Loss module."""
import torch
from torch import nn


def wgan_fake_loss(input: torch.Tensor) -> torch.Tensor:
    """Compute the Wasserstein loss for discriminator.

    Args:
        input (torch.Tensor): input tensor.

    Returns:
        torch.Tensor:
    """
    return torch.mean(input)


def wgan_real_loss(input: torch.Tensor) -> torch.Tensor:
    """Compute the Wasserstein loss for generator.

    Args:
        input (torch.Tensor): input tensor.

    Returns:
        torch.Tensor:
    """
    return -torch.mean(input)


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
) -> torch.Tensor:
    """Compute the gradient penalty.

    Args:
        discriminator (nn.Module): _description_
        real_images (torch.Tensor): _description_
        generated_images (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    t = torch.rand(real_images.size(0), dtype=real_images.dtype).to(real_images.device)
    t = t.expand_as(real_images.T).T

    # interpolation
    mid = t * real_images + (1 - t) * generated_images
    # set it to require grad info
    mid.requires_grad_()
    pred = discriminator(mid)

    grads = torch.autograd.grad(
        outputs=pred,
        inputs=mid,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # TODO: see the dims
    gp = torch.pow(torch.linalg.norm(torch.flatten(grads, 1), 2, dim=-1) - 1, 2).mean()

    return gp

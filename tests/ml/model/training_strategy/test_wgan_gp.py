import pytest
import torch
from torch import nn

from comic_cover_generator.ml.model.training_strategy.wgan_gp import (
    critic_loss_fn,
    generator_loss_fn,
    gradient_penalty,
)


@pytest.fixture(scope="module")
def critic():
    return nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))


@pytest.fixture(scope="module")
def real_imgs():
    return torch.rand(10, 2, requires_grad=True)


@pytest.fixture(scope="module")
def fake_imgs():
    return torch.rand(10, 2, requires_grad=True)


@pytest.mark.parametrize(
    "input, correct", [[1, 1], [[1, 1], 1], [[1, 2], 1.5], [list(range(10)), 4.5]]
)
def test_generator_loss_fn_pass(input, correct):
    assert generator_loss_fn(torch.as_tensor(input, dtype=torch.float)) == -1 * correct


def test_gradient_penalty_pass(real_imgs, fake_imgs, critic):
    gradient_penalty(critic, real_imgs.data, fake_imgs.data)


def test_critic_loss_pass(real_imgs, fake_imgs, critic):
    real_pred = critic(real_imgs)
    fake_pred = critic(fake_imgs)
    opt = torch.optim.SGD(critic.parameters(), lr=0.1)
    opt.zero_grad()
    gp = gradient_penalty(critic, real_imgs.data, fake_imgs.data)
    critic_loss_fn(real_pred, fake_pred, gp, 0.1).backward()
    opt.step()

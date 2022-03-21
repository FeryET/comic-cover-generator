import pytest
import torch
from torch import nn

from comic_cover_generator.ml.loss import (
    compute_gradient_penalty,
    wgan_fake_loss,
    wgan_real_loss,
)


@pytest.fixture(scope="module")
def discriminator():
    return nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))


@pytest.fixture(scope="module")
def real_imgs():
    return torch.rand(10, 2)


@pytest.fixture(scope="module")
def fake_imgs():
    return torch.rand(10, 2)


@pytest.mark.parametrize(
    "input, correct", [[1, 1], [[1, 1], 1], [[1, 2], 1.5], [list(range(10)), 4.5]]
)
def test_wgan_fake_loss(input, correct):
    assert wgan_fake_loss(torch.as_tensor(input, dtype=torch.float)) == correct


@pytest.mark.parametrize(
    "input, correct", [[1, -1], [[1, 1], -1], [[1, 2], -1.5], [list(range(10)), -4.5]]
)
def test_wgan_real_loss(input, correct):
    assert wgan_real_loss(torch.as_tensor(input, dtype=torch.float)) == correct


def test_gradient_penalty_pass(real_imgs, fake_imgs, discriminator):
    compute_gradient_penalty(discriminator, real_imgs, fake_imgs)

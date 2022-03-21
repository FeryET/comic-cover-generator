from cmath import isclose

import pytest
import torch
from torch.nn import functional as F

from comic_cover_generator.ml.model import Discriminator, wgan_fake_loss


@pytest.fixture(scope="module")
def disc():
    return Discriminator()


@pytest.fixture(scope="module")
def disc_input(disc):
    return torch.rand(1, 3, *disc.input_shape, requires_grad=False)


@torch.no_grad()
def test_discriminator_forward_pass(disc, disc_input):
    disc(disc_input)


@torch.no_grad()
def test_discriminator_output_shape(disc: Discriminator, disc_input):
    assert disc(disc_input).size()[-1] == 1


@pytest.mark.training
def test_discriminator_overfitting(
    disc: Discriminator, torch_random_generator: torch.Generator
):
    batch = torch.rand(2, 3, *disc.input_shape, generator=torch_random_generator)
    opt = torch.optim.AdamW(params=disc.parameters())
    target = torch.ones(2, 1, dtype=torch.float32)
    for _ in range(10):
        opt.zero_grad()
        output = disc(batch)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        opt.step()
    print(loss)
    assert isclose(loss, 0, abs_tol=0.01)

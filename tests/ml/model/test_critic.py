import numpy as np
import pytest
import torch
from torch.nn import functional as F

from comic_cover_generator.ml.model import Critic


@pytest.fixture(scope="module")
def disc():
    return Critic()


@pytest.fixture(scope="module")
def disc_input(disc):
    return torch.rand(1, 3, *disc.input_shape, requires_grad=False)


@torch.no_grad()
def test_critic_forward_pass(disc, disc_input):
    disc(disc_input)


@torch.no_grad()
def test_critic_output_shape(disc: Critic, disc_input):
    assert disc(disc_input).size()[-1] == 1


@pytest.mark.training
def test_critic_overfitting(disc: Critic, torch_random_generator: torch.Generator):
    batch = torch.rand(2, 3, *disc.input_shape, generator=torch_random_generator)
    opt = torch.optim.AdamW(params=disc.parameters(), lr=0.01)
    target = torch.ones(2, 1, dtype=torch.float32)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        output = F.sigmoid(disc(batch))
        l = F.binary_cross_entropy(output, target)
        l.backward()
        opt.step()
        losses.append(l.detach().item())
    assert np.isclose(losses[-1], 0, atol=0.01)

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from comic_cover_generator.ml.model import Critic


@pytest.fixture(scope="module")
def disc():
    return Critic(channels=[512, 512], input_shape=(8, 8))


@pytest.fixture(scope="module")
def disc_input(disc):
    return torch.rand(1, 3, *disc.input_shape)


@torch.no_grad()
def test_critic_from_rgb_forward_pass(disc: Critic, disc_input):
    disc.from_rgb(disc_input)


@torch.no_grad()
def test_critic_features_forward_pass(disc: Critic):
    disc.features(torch.rand(1, disc.channels[0], *disc.input_shape))


@torch.no_grad()
def test_critic_clf_forward_pass(disc: Critic):
    disc.clf(torch.rand(1, disc.channels[-1], *disc.input_shape))


@torch.no_grad()
def test_critic_forward_pass(disc, disc_input):
    disc(disc_input)


@torch.no_grad()
def test_critic_output_shape(disc: Critic, disc_input):
    assert disc(disc_input).size()[-1] == 1


@pytest.mark.training
def test_critic_overfitting(disc: Critic, torch_random_generator: torch.Generator):
    batch = torch.rand(1, 3, *disc.input_shape, generator=torch_random_generator)
    opt = torch.optim.AdamW(params=disc.parameters(), lr=0.1)
    target = torch.ones(1, 1, dtype=torch.float32)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        output = F.sigmoid(disc(batch))
        l = F.binary_cross_entropy(output, target)
        l.backward()
        opt.step()
        losses.append(l.detach().item())
    assert np.isclose(losses[-1], 0, atol=0.01)

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from comic_cover_generator.ml.model import Critic

torch.manual_seed(42)


@pytest.fixture(scope="module")
def disc():
    return Critic(channels=[8, 8, 8, 8, 8], input_shape=(64, 64))


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
    disc.clf(torch.rand(1, disc.channels[-1], 4, 4))


@torch.no_grad()
def test_critic_forward_pass(disc, disc_input):
    disc(disc_input)


@torch.no_grad()
def test_critic_output_shape(disc: Critic, disc_input):
    assert disc(disc_input).size()[-1] == 1


@pytest.mark.training
def test_critic_overfitting(disc: Critic):
    batch = torch.ones(8, 3, *disc.input_shape) * 0.5
    opt = torch.optim.AdamW(params=disc.parameters(), lr=0.02, eps=1e-8, weight_decay=0)
    target = torch.ones(batch.size(0), 1, dtype=torch.float32)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        output = disc(batch)
        cur_loss = F.binary_cross_entropy_with_logits(output, target)
        cur_loss.backward()
        opt.step()
        losses.append(cur_loss.detach().item())
    assert np.isclose(losses[-1], 0, atol=0.01)

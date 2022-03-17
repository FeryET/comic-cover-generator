from unittest import mock

import pytest
import torch
from torch import nn

from comic_cover_generator.ml.model import GAN


class MockedGen(nn.Module):
    latent_dim: int = 10

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(self.latent_dim, 100)

    def forward(self, x):
        return self.linear(x)


class MockedDisc(nn.Module):
    input_shape: int = 100

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(self.input_shape, 1)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture(scope="module")
def model():
    with mock.patch("comic_cover_generator.ml.model.Generator") as gen_mock:
        with mock.patch("comic_cover_generator.ml.model.Discriminator") as disc_mock:
            gen_mock.return_value = MockedGen()
            disc_mock.return_value = MockedDisc()
            yield GAN(pretrained=False)


@torch.no_grad()
def test_training_step_input_pass(model):
    # pytest.set_trace()
    batch = {"image": torch.rand(1, MockedDisc.input_shape)}
    model.training_step(batch, 0, 0)
    model.training_step(batch, 0, 1)

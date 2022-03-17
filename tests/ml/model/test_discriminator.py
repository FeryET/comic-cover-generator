import pytest
import torch

from comic_cover_generator.ml.model import Discriminator


@pytest.fixture(scope="module")
def disc():
    return Discriminator(pretrained=False)


@pytest.fixture(scope="module")
def disc_input(disc):
    return torch.rand(1, 3, *disc.input_shape, requires_grad=False)


@torch.no_grad()
def test_discriminator_forward_pass(disc, disc_input):
    disc(disc_input)


@torch.no_grad()
def test_discriminator_output_shape(disc: Discriminator, disc_input):
    assert disc(disc_input).size()[-1] == 1

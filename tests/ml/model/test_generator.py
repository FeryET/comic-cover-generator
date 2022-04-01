import random
from unittest import mock

import pytest
import torch

from comic_cover_generator.ml.model import Critic, Generator


@pytest.fixture(scope="module")
def gen():
    return Generator(
        16,
        16,
        (8, 4, 2),
        char_cnn_channels=(8, 8),
        output_shape=(16, 16),
    )


@torch.no_grad()
def test_generator_output_shape_match_critic_input_shape(gen: Generator):
    assert (
        gen.forward(
            torch.rand(1, gen.latent_dim),
            [torch.randint(0, 256, [random.randint(5, 10)])],
        ).size()[-2:]
        == gen.output_shape
    )

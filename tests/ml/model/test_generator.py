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
        output_shape=(16, 16),
        sequence_gru_hidden_size=16,
        sequence_embed_dim=4,
        sequence_gru_layers=1,
    )


@torch.no_grad()
def test_generator_output_shape_match_critic_input_shape(gen: Generator):
    assert (
        gen.forward(
            torch.rand(1, gen.latent_dim),
            [torch.randint(0, 256, [random.randint(1, 3)])],
        ).size()[-2:]
        == gen.output_shape
    )

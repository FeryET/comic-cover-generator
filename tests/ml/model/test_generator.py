import random
from unittest import mock

import pytest
import torch

from comic_cover_generator.ml.model import Critic, Generator


class MockedPGAN:
    def __init__(self):
        pass

    def getNetG(self):
        def features(x: torch.Tensor, feature_size=Generator.output_shape):
            if x.size()[-1] != Generator.latent_dim:
                raise RuntimeError("Mismatch of input shape.")
            return torch.rand(len(x), 3, *feature_size)

        return features


@pytest.fixture(scope="module")
def gen():
    with mock.patch("torch.hub.load") as mocked:
        mocked.return_value = MockedPGAN()
        yield Generator()


@torch.no_grad()
def test_generator_output_shape_match_critic_input_shape(gen: Generator):
    assert (
        gen.forward(
            torch.rand(1, gen.latent_dim),
            [torch.randint(0, 256, [random.randint(1, 3)])],
            torch.rand(1, *gen.output_shape),
        ).size()[-2:]
        == Critic.input_shape
    )

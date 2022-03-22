from unittest import mock

import pytest
import torch

from comic_cover_generator.ml.model import Critic, Generator


class MockedPGAN:
    def __init__(self):
        pass

    def getNetG(self):
        def features(x: torch.Tensor, feature_size=(256, 256)):
            if x.size()[-1] != Generator.latent_dim:
                raise RuntimeError("Mismatch of input shape.")
            return torch.rand(len(x), 3, *feature_size)

        return features


@pytest.fixture(scope="module")
def gen():
    with mock.patch("torch.hub.load") as mocked:
        mocked.return_value = MockedPGAN()
        yield Generator(pretrained=False)


@torch.no_grad()
def test_generator_resizer_shape_match_critic_input_shape(gen):
    assert gen(torch.rand(1, gen.latent_dim)).size()[-2:] == Critic.input_shape

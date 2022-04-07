import random
from unittest import mock

import pytest
import torch

from comic_cover_generator.ml.model import Generator


class MockedSeq2Vec(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hidden_size = 8

    def forward(self, seq):
        return torch.ones(len(seq["input_ids"]), self.hidden_size)


@pytest.fixture(scope="module")
def gen():
    with mock.patch("comic_cover_generator.ml.model.generator.Seq2Vec", MockedSeq2Vec):
        return Generator(
            16,
            16,
            (8, 4, 2),
            output_shape=(16, 16),
        )


@pytest.fixture
def batch():
    return torch.rand(2, 16), {"input_ids": [[1, 2, 3, 1, 0], [1, 0, 0, 1, 0]]}


@torch.no_grad()
def test_generator_output_shape_match_critic_input_shape(batch, gen: Generator):
    assert gen.forward(*batch).size()[-2:] == gen.output_shape

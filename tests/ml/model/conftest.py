import pytest
import torch

from tests.fixtures import image_size, seed


@pytest.fixture(scope="session")
def torch_random_generator(seed):
    return torch.Generator().manual_seed(seed)

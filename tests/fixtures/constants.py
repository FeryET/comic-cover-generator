from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def seed():
    return 42


@pytest.fixture(scope="session", autouse=True)
def test_cache_folder(tmpdir_factory):
    folder_path = Path(str(tmpdir_factory.mktemp("test_caches")))
    return folder_path


@pytest.fixture(scope="session", autouse=True)
def random_state(seed):
    return np.random.default_rng(seed)


@pytest.fixture(scope="session", autouse=True)
def image_size():
    return (184, 128)

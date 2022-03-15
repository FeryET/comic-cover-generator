import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def generated_data_folder(tmp_path):
    folder_path = Path(tmp_path) / "generated_dataset"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture
def metadata_csv(generated_data_folder):
    return generated_data_folder / "metadata.csv"


@pytest.fixture
def images_folder(generated_data_folder: Path):
    folder = generated_data_folder / "images_folder"
    folder.mkdir()
    return folder


@pytest.fixture
def random_seed():
    return np.random.default_rng(42)


@pytest.fixture
def image_size():
    return (250, 250)


@pytest.fixture
def metadata_dataframe(metadata_csv, images_folder, random_seed, image_size):
    metadata = {
        "series": [
            "first",
            "second",
            "third",
            "fourth",
        ],
        "fulltitle": [
            "first full title",
            "second full title",
            "third full title",
            "fourth full title",
        ],
        "image_url": [
            "/image/first.jpg",
            "/image/second.jpg",
            "/image/third.jpg",
            "/image/fourth.jpg",
        ],
    }
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_csv, index_label=False)

    for row in df.itertuples():
        img = random_seed.integers(0, 256, (*image_size, 3)).astype(np.uint8)
        path = str(Path(images_folder) / row.image_url[7:])
        Image.fromarray(img).save(path)

    return df


def test_setup_data_works(metadata_dataframe):
    pass

import os
import string
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
        "series": [str(i) for i in range(11, 30)],
    }
    df = pd.DataFrame(metadata)
    df["fulltitle"] = df["series"].apply(lambda x: f"{x} full title")
    df["image_url"] = df["series"].apply(lambda x: f"/image/{x}.jpg")
    df["image_path"] = df["image_url"].apply(lambda x: f"{images_folder}/{x[7:]}")
    df.to_csv(metadata_csv, index_label=False)

    for row in df.itertuples():
        img = random_seed.integers(0, 256, (*image_size, 3)).astype(np.uint8)
        Image.fromarray(img).save(row.image_path)

    return df


def test_setup_data_works(metadata_dataframe):
    pass

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tests.fixtures import image_size, random_state, seed, test_cache_folder


@pytest.fixture(scope="session", autouse=True)
def generated_data_folder(tmpdir_factory):
    folder_path = Path(str(tmpdir_factory.mktemp("data"))) / "generated_dataset"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture(scope="session", autouse=True)
def metadata_csv(generated_data_folder):
    return generated_data_folder / "metadata.csv"


@pytest.fixture(scope="session", autouse=True)
def images_folder(generated_data_folder: Path):
    folder = generated_data_folder / "images_folder"
    folder.mkdir(exist_ok=True)
    return folder


@pytest.fixture(scope="session")
def metadata_dataframe(metadata_csv, images_folder, random_state, image_size):
    metadata = {
        "series": [str(i) for i in range(11, 30)],
    }
    df = pd.DataFrame(metadata)
    df["full_title"] = df["series"].apply(lambda x: f"{x} full title")
    df["img_url"] = df["series"].apply(lambda x: f"/image/{x}.jpg")
    df["image_path"] = df["img_url"].apply(lambda x: f"{images_folder}/{x[7:]}")
    df.to_csv(metadata_csv, index_label=False)

    for row in df.itertuples():
        curr_img_size = random_state.integers(10, image_size, len(image_size))
        img = random_state.integers(0, 256, (*curr_img_size, 3)).astype(np.uint8)
        Image.fromarray(img).save(row.image_path)

    return df


def test_setup_data_works(metadata_dataframe):
    pass

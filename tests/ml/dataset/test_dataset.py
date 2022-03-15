import os
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pandas as pd
import pytest

from comic_cover_generator.ml import dataset


@pytest.fixture(
    scope="function",
    params=[
        #
        lambda x: int(str(x)[-1]) % 2 == 0,
        lambda x: hash(x) % 2 == 0,
        lambda x: str(x).startswith("f"),
    ],
)
def path_exists_mocker(request):
    return mock.patch.object(
        Path,
        "exists",
        request.param,
    )


@pytest.fixture
def filenames(tmp_path, path_exists_mocker):
    return [
        str(Path(tmp_path) / item)
        for item in ["filename1", "milename2", "filename3", "filename4"]
    ]


def test_filter_non_existant_images_pass(filenames, path_exists_mocker):
    df = pd.DataFrame({"image_path": filenames, "repeated": filenames})
    df = dataset._filter_non_existant_images(df)
    assert all(df["image_path"].apply(lambda x: Path(x).exists()))


def test_create_image_path_column_pass(metadata_dataframe, images_folder):
    assert all(
        dataset._create_image_path_column(metadata_dataframe, images_folder)[
            "image_path"
        ].apply(os.path.exists)
    )


def test_cover_dataset_create_hdf5_pass(tmp_path, metadata_dataframe, image_size):
    dataset.CoverDataset.create_hdf5_image_dataset(
        metadata_dataframe["image_path"],
        os.path.join(tmp_path, "temporary.h5"),
        image_size,
    )

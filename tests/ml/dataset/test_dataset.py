from pathlib import Path
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
    autouse=True,
)
def path_exists_mocker(mocker, request):
    return mocker.patch.object(
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


def test_filter_non_existant_images(filenames, path_exists_mocker):
    df = pd.DataFrame({"image_path": filenames, "repeated": filenames})
    print("###########", path_exists_mocker)
    df = dataset._filter_non_existant_images(df)
    assert all(df["image_path"].apply(lambda x: Path(x).exists()))

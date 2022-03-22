import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from torchvision import transforms as vision_transforms

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
def filenames(tmp_path):
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


class TestCoverDatasetClassMethods:
    def test_cover_dataset_create_hdf5_pass(
        self, metadata_dataframe, tmp_path, image_size
    ):
        dataset.CoverDataset.create_hdf5_image_dataset(
            metadata_dataframe["image_path"],
            os.path.join(tmp_path, "temporary.h5"),
            image_size,
        )


class TestCoverDatasetInstanceMethods:
    @pytest.fixture(scope="class", params=[True, False])
    def preload_images(self, request):
        return request.param

    @pytest.fixture(
        scope="class",
        params=[
            None,
            vision_transforms.Compose(
                [
                    vision_transforms.ToTensor(),
                ]
            ),
            vision_transforms.Compose(
                [vision_transforms.ColorJitter(), vision_transforms.ToTensor()]
            ),
        ],
    )
    def image_transforms(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def dataset_instance(
        self,
        test_cache_folder,
        preload_images,
        image_transforms,
        metadata_csv,
        images_folder,
        image_size,
    ):
        return dataset.CoverDataset(
            metadata_csv=metadata_csv,
            images_folder=images_folder,
            preload_images=preload_images,
            preload_path=os.path.join(test_cache_folder, "dataset_cache"),
            image_size=image_size,
            image_transforms=image_transforms,
        )

    def test_cover_dataset_length(
        self,
        dataset_instance,
        metadata_dataframe,
    ):
        assert len(dataset_instance) == len(metadata_dataframe)

    def test_cover_dataset_iteration(self, dataset_instance):
        for _ in dataset_instance:
            pass

    def test_cover_dataset_image_shape_pass(self, dataset_instance, image_size):
        image = dataset_instance[0]["image"]
        assert image.size() == (3, *image_size)

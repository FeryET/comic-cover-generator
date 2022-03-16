"""Dataset module."""
import platform
from collections import namedtuple
from pathlib import Path
from typing import Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as vision_transforms
from tqdm.auto import tqdm

if float(platform.sys.version[:3]) < 3.8:
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


def _create_image_path_column(
    metadata_df: pd.DataFrame, images_folder: str
) -> pd.DataFrame:
    metadata_df["image_path"] = metadata_df["image_url"].apply(
        lambda x: str(Path(images_folder).joinpath(*Path(x).parts[2:]))
    )
    return metadata_df


def _filter_non_existant_images(metadata_df: pd.DataFrame) -> pd.DataFrame:
    return metadata_df.iloc[
        metadata_df["image_path"].apply(lambda x: Path(x).exists()).values
    ].reset_index(drop=True)


def _read_resized_image(path: str, image_size: int) -> Image.Image:
    return ImageOps.pad(
        Image.open(path).convert("RGB"),
        image_size,
        color=0,
        centering=(0.5, 0.5),
    )


class CoverDatasetItem(TypedDict):
    """An item returned by the py:func:`CoverDataset<comic_cover_generator.ml.dataset.CoverDataet.__getitem__()`."""

    image: Tensor


class CoverDataset(Dataset):
    """The dataset for cover images."""

    Data = namedtuple("CoverDatasetData", ["images", "metadata"])

    @classmethod
    def create_hdf5_image_dataset(
        cls,
        image_paths: Sequence[str],
        h5_file_path: str,
        image_size: Tuple[int, int],
    ) -> h5py.Dataset:
        """Create hd5 image dataset given the inputs.

        Args:
            image_paths (Sequence[str]): Paths of the images to load.
            h5_file_path (str): Path to save the dataset in.
            image_size (Tuple[int, int]): what should the image size be.

        Returns:
            h5py.Dataset: _description_
        """
        h5_file = h5py.File(h5_file_path, "a")
        if (
            # check if the dataset doest not exists
            "images" not in h5_file
            # or if the dataset has a different shape
            or h5_file["images"].shape[2:] != image_size
            # or if the dataset creation was terminated mid-creation
            or h5_file["images"].attrs.get("status") != "ready"
        ):
            # clear the file anyway
            h5_file.clear()
            dataset = h5_file.create_dataset(
                "images",
                shape=(len(image_paths), *image_size, 3),
                dtype=np.uint8,
                chunks=(5, *image_size, 3),
                compression="gzip",
            )
            for index, fpath in enumerate(
                tqdm(image_paths, desc="creating images hdf5 dataset")
            ):
                dataset[index] = np.asarray(
                    _read_resized_image(fpath, image_size),
                    dtype=np.uint8,
                )
            dataset.attrs["status"] = "ready"
        else:
            dataset = h5_file["images"]
        return dataset

    def __init__(
        self,
        metadata_csv: str,
        images_folder: str,
        preload_images=True,
        preload_path="cache/",
        image_size: Tuple[int, int] = (250, 250),
        image_transforms: vision_transforms.Compose = None,
    ) -> None:
        """Initialize an instance of CoverDataset.

        Args:
            metadata_csv (str): The path to csv file containing metadata.
            images_folder (str): The path to the images folder.
            preload_images (bool, optional): Whether to preload images in h5 format. Defaults to True.
            preload_path (str, optional): The path to preload images. Defaults to "cache/".
            image_size (Tuple[int, int], optional): Image size. Defaults to (250, 250).
            image_transforms (torchvision.transforms.Compose): Image transforms.
        """
        self.image_folder = Path(images_folder)
        metadata_df = _filter_non_existant_images(
            _create_image_path_column(pd.read_csv(metadata_csv), images_folder)
        )
        self.preload_images = preload_images
        self.image_size = image_size

        with pd.option_context("display.max_columns", None):
            print(metadata_df)
        self.data = CoverDataset.Data(
            images=(
                CoverDataset.create_hdf5_image_dataset(
                    metadata_df["image_path"], preload_path, image_size
                )
                if self.preload_images
                else metadata_df["image_path"]
            ),
            metadata=metadata_df,
        )

        if image_transforms is None:
            self.image_transforms = vision_transforms.Compose(
                vision_transforms.ToTensor()
            )
        elif not any(
            isinstance(trns, vision_transforms.ToTensor)
            for trns in image_transforms.transforms
        ):
            raise ValueError(
                "Please add a ToTensor trnasformation to the Compose object."
            )

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int:
        """
        return len(self.data["metadata"])

    def __getitem__(self, index: int) -> CoverDatasetItem:
        """Get the item in dataset.

        Args:
            index (int):

        Returns:
            CoverDatasetItem: A dictionary of tensors.
        """
        if self.preload_images:
            image = self.data.images[index]
        else:
            # read PIL image
            image = _read_resized_image(self.data.images[index])
        image = self.image_transforms(image)
        return {"image": image}

"""Dataset module."""
import string
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageOps
from pytorch_lightning import LightningDataModule
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms as vision_transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.typing import TypedDict


def _create_image_path_column(
    metadata_df: pd.DataFrame, images_folder: str
) -> pd.DataFrame:
    metadata_df["image_path"] = metadata_df["img_url"].apply(
        lambda x: str(Path(images_folder).joinpath(*Path(x).parts[2:]))
    )
    return metadata_df


def _filter_non_existant_images(metadata_df: pd.DataFrame) -> pd.DataFrame:
    return metadata_df.iloc[
        metadata_df["image_path"].apply(lambda x: Path(x).exists()).values
    ].reset_index(drop=True)


def _pad_image_to_shape(image: Image.Image, image_size: Tuple[int, int]) -> Image.Image:
    return ImageOps.pad(
        image,
        image_size[::-1],
        color=0,
        centering=(0.0, 0.0),
    )


def _resize_image_to_shape(
    image: Image.Image, image_size: Tuple[int, int]
) -> Image.Image:
    return image.resize(image_size[::-1], resample=Image.NEAREST)


GOOD_CHARS = set(string.printable) - set(string.whitespace)


class MapToMinusOneAndOne:
    """Transforms which maps a tensor from [0,1] to [-1, 1]."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Mapped tensor.
        """
        return (2 * x) - 1


class CoverDatasetItem(TypedDict):
    """An item returned by the py:func:`CoverDataset<comic_cover_generator.ml.dataset.CoverDataet.__getitem__()`."""

    image: Tensor
    full_title: str


class CoverDatasetCollater:
    """Batch collater function for cover dataset."""

    class CoverDatasetBatch(TypedDict):
        """Batch item for cover dataset."""

        image: torch.Tensor
        title_seq: BatchEncoding

    def __init__(self, tokenizer: Tokenizer, max_length: int) -> None:
        """Initialize a collator.

        Args:
            tokenizer (Tokenizer):
            max_length (int):

        Returns
            None:
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self, batch: Sequence[CoverDatasetItem]
    ) -> "CoverDatasetCollater.CoverDatasetBatch":
        """Collate a batch.

        Args:
            batch (Sequence[CoverDatasetItem]):

        Returns:
            CoverDatasetCollater.CoverDatasetBatch:
        """
        images = torch.stack([b["image"] for b in batch])

        title_seq = self.tokenizer(
            [b["full_title"] for b in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {"image": images, "title_seq": title_seq}


class CoverDatasetParams(TypedDict):
    """Parameter set for dataset."""

    metadata_csv: str
    images_folder: str
    preload_images: bool
    preload_path: str
    image_size: Tuple[int, int]
    image_transforms: vision_transforms.Compose


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
            h5py.Dataset: The returned dataset.
        """
        h5_file = h5py.File(h5_file_path, "a")
        if (
            # check if the dataset doest not exists
            "images" not in h5_file
            # or if the dataset has a different shape
            # shape is N, H, W, C -> H, W = shape[1:-1]
            or all(h5_file["images"].shape[1:-1] != np.asarray(image_size))
            # or if the dataset creation was terminated mid-creation
            or h5_file["images"].attrs.get("status") != "ready"
        ):
            # clear the file anyway
            h5_file.clear()
            dataset = h5_file.create_dataset(
                "images",
                shape=(len(image_paths), *image_size, 3),
                dtype=np.uint8,
                chunks=(10, *image_size, 3),
                compression="gzip",
            )
            for index, fpath in enumerate(
                tqdm(image_paths, desc="creating images hdf5 dataset")
            ):
                image = Image.open(fpath).convert("RGB")
                dataset[index] = np.asarray(
                    _resize_image_to_shape(image, image_size),
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
        preload_images: bool = True,
        preload_path: str = "cache/",
        image_size: Tuple[int, int] = (92, 64),
        image_transforms: vision_transforms.Compose = None,
    ) -> None:
        """Initialize an instance of CoverDataset.

        Args:
            metadata_csv (str): The path to csv file containing metadata.
            images_folder (str): The path to the images folder.
            preload_images (bool, optional): Whether to preload images in h5 format. Defaults to True.
            preload_path (str, optional): The path to preload images. Defaults to "cache/".
            image_size (Tuple[int, int], optional): Image size. Defaults to (128, 184).
            image_transforms (torchvision.transforms.Compose): Image transforms.
        """
        self.image_folder = Path(images_folder)
        self.preload_images = preload_images
        self.image_size = image_size

        Path(preload_path).mkdir(exist_ok=True)

        metadata_df = _filter_non_existant_images(
            _create_image_path_column(pd.read_csv(metadata_csv), images_folder)
        )

        h5_file_path = str(
            Path(preload_path) / ("cache_" + Path(metadata_csv).stem + ".h5")
        )

        self.data = CoverDataset.Data(
            images=(
                CoverDataset.create_hdf5_image_dataset(
                    metadata_df["image_path"].values,
                    h5_file_path,
                    image_size,
                )
                if self.preload_images
                else metadata_df["image_path"].values
            ),
            metadata=metadata_df,
        )

        if len(self.data.metadata) != len(self.data.images):
            raise RuntimeError(
                "metadata dataframe and image_paths are not equal length!"
            )

        if image_transforms is None:
            self.image_transforms = vision_transforms.Compose(
                [
                    vision_transforms.ToTensor(),
                ]
            )
        elif not any(
            isinstance(trns, vision_transforms.ToTensor)
            for trns in image_transforms.transforms
        ):
            raise ValueError(
                "Please add a ToTensor trnasformation to the Compose object."
            )
        else:
            self.image_transforms = image_transforms

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int:
        """
        return len(self.data.metadata)

    def __getitem__(self, index: int) -> CoverDatasetItem:
        """Get the item in dataset.

        Args:
            index (int):

        Returns:
            CoverDatasetItem: A dictionary of tensors.
        """
        if self.preload_images:
            image = Image.fromarray(self.data.images[index])
        else:
            # read PIL image
            image = Image.open(self.data.images[index]).convert("RGB")
            image = _resize_image_to_shape(image, self.image_size)

        image = self.image_transforms(image)
        full_title = self.data.metadata.iloc[index]["full_title"]

        return {"image": image, "full_title": full_title}


class CoverDataModule(LightningDataModule):
    """Cover data module for encapsulating cover dataset."""

    class Subsets(NamedTuple):
        """Subsets of a dataset."""

        train: Subset
        val: Subset

    class SubsetLengths(NamedTuple):
        """Split lengths of a dataset."""

        train: float
        val: float

    def __init__(
        self,
        dataset_params: CoverDatasetParams,
        batch_size: int,
        max_seq_length: 30,
        transformer_model: str = "prajjwal1/bert-tiny",
        subsets_lengths: "CoverDataModule.SubsetLengths" = None,
        random_seed: int = None,
    ):
        """Initialize a data module.

        Args:
            dataset (CoverDataset):
            batch_size (int):
            subsets_lengths (CoverDataModule.Splits, optional): Defaults to None.
            random_seed (int, optional): Defaults to None.

        Returns:
            None:
        """
        super().__init__()
        self.dataset = CoverDataset(**dataset_params)
        self.batch_size = batch_size
        self.random_generator = torch.Generator("cpu").manual_seed(random_seed)
        self.subsets: CoverDataModule.Subsets = None

        train_length = int(
            len(self.dataset) * subsets_lengths[0] / sum(subsets_lengths)
        )
        self.subsets_lengths = CoverDataModule.SubsetLengths(
            train_length, len(self.dataset) - train_length
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model, cache_dir=Constants.cache_dir
        )

        self.max_seq_length = max_seq_length

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the data module.

        Args:
            collate_fn (Callable):
            stage (Optional[str], optional): Defaults to None.

        Returns:
            None:
        """
        self.subsets = CoverDataModule.Subsets(
            *random_split(
                self.dataset, self.subsets_lengths, generator=self.random_generator
            )
        )

        self.collate_fn = CoverDatasetCollater(self.tokenizer, self.max_seq_length)

    def prepare_data(self) -> None:
        """Does nothing."""
        super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        """Get the train dataloader.

        Returns:
            DataLoader:
        """
        return DataLoader(
            self.subsets.train,
            self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
            num_workers=psutil.cpu_count(logical=False),
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns:
            DataLoader:
        """
        return DataLoader(
            self.subsets.val,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            num_workers=psutil.cpu_count(logical=False),
        )

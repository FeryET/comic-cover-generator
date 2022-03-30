"""Training script."""

import hydra
import mlflow
import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.dataset import CoverDataset
from comic_cover_generator.ml.model import GAN

CONFIG_PATH = "../../conf/train/"


def generate_training_images_grid(
    dataset: CoverDataset, shape=(4, 4), random_seed: int = 42
) -> None:
    """Generate a grid for training images.

    Args:
        dataset (CoverDataset): Input dataset.
        shape (tuple, optional): Defaults to (4, 4).
        random_seed (int, optional): Defaults to 42.
    """
    import numpy as np
    import torch

    from comic_cover_generator.ml.utils import make_captioned_grid

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(np.arange(len(dataset)), size=shape, replace=False)

    images, titles = [], []

    for idx in indices.flat:
        item = dataset[idx]
        images.append(item["image"])
        titles.append(item["full_title"])

    images = torch.stack(images)

    make_captioned_grid(
        images,
        titles,
        shape,
        value_range=(-1, 1),
        normalize=True,
        figsize=(16, 16),
        fontsize=12,
        fname="training_image_samples",
    )


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg (DictConfig): The hydra configuration object.
    """
    OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))

    config = instantiate(cfg, _convert_="partial")

    is_float16 = config["trainer"].get("precision", 32)

    # fix epsilon
    Constants.eps = config.get("eps", 1e-7 if is_float16 else 1e-8)

    # init data
    dataset = CoverDataset(**config["dataset"])
    generate_training_images_grid(dataset)

    # init model
    model = GAN(
        training_strategy_params=config["training_strategy_params"], **config["model"]
    )
    model.attach_train_dataset_and_generate_validtaion_data(dataset)

    # init trainer
    trainer = pl.Trainer(**config["trainer"])

    # init logger
    mlflow.set_tracking_uri(config["logger"].pop("tracking_uri"))
    mlflow.set_experiment("comic-cover-generator")
    mlflow.pytorch.autolog(**config["logger"])

    with mlflow.start_run():
        mlflow.log_artifact(".hydra/config.yaml", "config.yaml")
        trainer.tune(model)
        trainer.fit(model)


if __name__ == "__main__":
    train()

"""Training script."""

import hydra
import mlflow
import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from comic_cover_generator.ml.dataset import CoverDataset
from comic_cover_generator.ml.model import GAN

CONFIG_PATH = "../../conf/train/"


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg (DictConfig): The hydra configuration object.
    """
    OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))

    config = instantiate(cfg, _convert_="partial")

    # init data
    dataset = CoverDataset(**config["dataset"])

    # init model
    model = GAN(**config["model"])
    model.attach_train_dataset(dataset)

    # init trainer
    trainer = pl.Trainer(**config["trainer"])

    # init logger
    mlflow.pytorch.autolog(**config["logger"])

    with mlflow.start_run():
        mlflow.log_artifact(".hydra/config.yaml", "config.yaml")
        trainer.tune(model)
        trainer.fit(model)


if __name__ == "__main__":
    train()

"""Training script."""

import hydra
import mlflow
import pytorch_lightning as pl
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

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

    # init model
    model = GAN(**config["model"])
    model.make_partially_trainable()

    # init data
    dataset = CoverDataset(**config["dataset"])
    dataloader = DataLoader(dataset=dataset, batch_size=model.batch_size, shuffle=True)

    # init trainer
    trainer = pl.Trainer(**config["trainer"])

    # init logger
    mlflow.pytorch.autolog(**config["logger"])

    with mlflow.start_run():
        mlflow.log_artifact(".hydra/config.yaml", "config.yaml")
        trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()

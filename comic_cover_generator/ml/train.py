"""Training script."""

import hydra
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from comic_cover_generator.ml.dataset import CoverDataset
from comic_cover_generator.ml.model import GAN


@hydra.main(config_path="../../conf/train/", config_name="train")
def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg (DictConfig): The hydra configuration object.
    """
    OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))

    config = instantiate(cfg, _convert_="partial")
    GAN(**config["model"])
    CoverDataset(**config["dataset"])


if __name__ == "__main__":
    train()

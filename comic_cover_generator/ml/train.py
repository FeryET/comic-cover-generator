"""Training script."""

import os
from pathlib import Path
from typing import Tuple

import hydra
import mlflow
import psutil
import torch
from hydra.utils import get_class, get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.dataset import (
    CoverDataset,
    CoverDatasetCollater,
    split_dataset_to_subsets,
)
from comic_cover_generator.ml.model import Critic, Generator
from comic_cover_generator.ml.model.training_strategy.nsgan import (
    ns_gan_disc_loss,
    ns_gan_gen_loss,
    nsgan_critic_train_loop,
    nsgan_generator_train_loop,
)

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
        value_range=(0, 1),
        normalize=True,
        figsize=(16, 16),
        fontsize=12,
        fname="training_image_samples",
    )


def generate_inputs(
    batch: CoverDatasetCollater.CoverDatasetBatch,
    device: torch.device,
    output_shape: Tuple[int, int],
    latent_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate inputs for generator and critic.

    Args:
        batch (CoverDatasetCollater.CoverDatasetBatch):
        device (torch.device):
        output_shape (Tuple[int, int]):
        latent_dim (int):

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    title_seq: torch.Tensor = batch["title_seq"].to(device)
    reals: torch.Tensor = batch["image"].to(device)
    z = reals.new_empty((len(reals), latent_dim), device=device).normal_()
    stochastic_noise = reals.new_empty((len(reals), *output_shape)).normal_()
    return reals, title_seq, z, stochastic_noise


def save_model(
    fpath: str,
    generator: Generator,
    g_opt: torch.optim.Optimizer,
    g_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    critic: Critic,
    c_opt: torch.optim.Optimizer,
    c_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    epoch: int,
) -> None:
    """Save a model.

    Args:
        fpath (str):
        generator (Generator):
        g_opt (torch.optim.Optimizer):
        g_scaler (torch.cuda.amp.grad_scaler.GradScaler):
        critic (Critic):
        c_opt (torch.optim.Optimizer):
        c_scaler (torch.cuda.amp.grad_scaler.GradScaler):
        epoch (int):

    Returns:
        None:
    """
    torch.save(
        {
            "generator": generator.state_dict(),
            "g_opt": g_opt.state_dict(),
            "g_scaler": g_scaler.state_dict(),
            "critic": critic.state_dict(),
            "c_opt": c_opt.state_dict(),
            "c_scaler": c_scaler.state_dict(),
            "epoch": epoch,
        },
        fpath,
    )


def load_saved_model(
    fpath: str,
    generator: Generator,
    g_opt: torch.optim.Optimizer,
    g_scaler: torch.cuda.amp.grad_scaler.GradScaler,
    critic: Critic,
    c_opt: torch.optim.Optimizer,
    c_scaler: torch.cuda.amp.grad_scaler.GradScaler,
) -> int:
    """Load a saved model.

    Args:
        fpath (str):
        generator (Generator):
        g_opt (torch.optim.Optimizer):
        g_scaler (torch.cuda.amp.grad_scaler.GradScaler):
        critic (Critic):
        c_opt (torch.optim.Optimizer):
        c_scaler (torch.cuda.amp.grad_scaler.GradScaler):

    Returns:
        int:
    """
    state_dict = torch.load(fpath)
    generator.load_state_dict(state_dict["generator"])
    g_opt.load_state_dict(state_dict["g_opt"])
    g_scaler.load_state_dict(state_dict["g_scaler"])
    critic.load_state_dict(state_dict["critic"])
    c_opt.load_state_dict(state_dict["c_opt"])
    c_scaler.load_state_dict(state_dict["c_scaler"])
    return state_dict.get("epoch", 0)


@hydra.main(config_path=CONFIG_PATH, config_name="config")
def train(cfg: DictConfig):
    """Train a model.

    Args:
        cfg (DictConfig): The hydra configuration object.
    """
    OmegaConf.register_new_resolver(name="get_cls", resolver=lambda cls: get_class(cls))

    config = instantiate(cfg, _convert_="partial")

    is_mixed_precision = config["training"].get("mixed_precision", False)

    # fix epsilon
    Constants.eps = config.get("eps", 1e-7 if is_mixed_precision else 1e-8)
    Constants.cache_dir = os.path.join(get_original_cwd(), Constants.cache_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---------------------------------------------------------------------------- #
    #                                 init_dataset                                 #
    # ---------------------------------------------------------------------------- #
    dataset = CoverDataset(**config["dataset"])
    train_dataset, val_dataset = split_dataset_to_subsets(
        dataset, **config["dataset_split"]
    )
    generate_training_images_grid(val_dataset)

    # ---------------------------------------------------------------------------- #
    #                                  init model                                  #
    # ---------------------------------------------------------------------------- #
    generator = Generator(**config["model"]["generator"]).to(device)
    critic = Critic(**config["model"]["critic"]).to(device)
    g_opt = generator.create_optimizers(**config["model"]["optimizer"]["generator"])
    c_opt = critic.create_optimizers(**config["model"]["optimizer"]["critic"])

    # ---------------------------------------------------------------------------- #
    #                             init training params                             #
    # ---------------------------------------------------------------------------- #
    training_config = config["training"]
    mlflow.set_tracking_uri(training_config["logger"]["tracking_uri"])
    mlflow.set_experiment(training_config["logger"]["experiment_name"])

    checkpoint_path = training_config.get(
        "checkpoint_path",
        os.path.join(get_original_cwd(), "checkpoints", "model.cpkt"),
    )
    Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)

    collater = CoverDatasetCollater(**training_config["collater"])
    train_dataloader = DataLoader(
        train_dataset,
        training_config["batch_size"],
        shuffle=True,
        num_workers=psutil.cpu_count(logical=True),
        collate_fn=collater,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        training_config["batch_size"],
        shuffle=True,
        collate_fn=collater,
        pin_memory=True,
    )

    fid_metric = FrechetInceptionDistance(feature=768)

    g_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=is_mixed_precision)
    c_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=is_mixed_precision)

    # metrics and loading
    best_fid = float("inf")
    if training_config.get("load_from_checkpoint", False) is not False:
        start_epoch = load_saved_model(
            training_config["load_from_checkpoint"],
            generator,
            g_opt,
            g_scaler,
            critic,
            c_opt,
            c_scaler,
        )
    else:
        start_epoch = 0
    # ---------------------------------------------------------------------------- #
    #                                start training                                #
    # ---------------------------------------------------------------------------- #
    with mlflow.start_run():
        for epoch in range(start_epoch, training_config["max_epochs"]):
            # ---------------------------------------------------------------------------- #
            #                           training section in epoch                          #
            # ---------------------------------------------------------------------------- #
            g_loss_mean = 0
            c_loss_mean = 0
            batch_idx = 0
            pbar = tqdm(
                train_dataloader, desc=f"training iteration {epoch}", leave=False
            )
            for batch_idx, batch in enumerate(pbar):
                cur_training_step = batch_idx + epoch * len(train_dataset)
                reals, title_seq, z, noise = generate_inputs(
                    batch, device, generator.output_shape, generator.latent_dim
                )
                #################
                # train generator
                #################
                g_opt.zero_grad()
                if training_config["strategy"]["name"] == "nsgan":
                    fakes, g_loss, path_lengths = nsgan_generator_train_loop(
                        generator=generator,
                        critic=critic,
                        g_scaler=g_scaler,
                        g_opt=g_opt,
                        z=z,
                        title_seq=title_seq,
                        noise=noise,
                        batch_idx=batch_idx,
                        autocast_enabled=is_mixed_precision,
                        grad_clip_val=training_config["grad_clip_val"],
                        pl_beta=training_config["strategy"]["params"]["pl_beta"],
                        pl_coef=training_config["strategy"]["params"]["pl_coef"],
                        pl_interval=training_config["strategy"]["params"][
                            "pl_interval"
                        ],
                    )

                fid_metric.update(generator.to_uint8(reals), real=True)
                fid_metric.update(generator.to_uint8(fakes), real=False)

                mlflow.log_metric("g_loss_train_step", g_loss, step=cur_training_step)
                if path_lengths is not None:
                    mlflow.log_metric(
                        "ppl_train_step",
                        path_lengths,
                        step=cur_training_step,
                    )
                g_loss_mean += g_loss
                ##############
                # train critic
                ##############
                c_opt.zero_grad()
                reals.requires_grad_(True)
                if training_config["strategy"]["name"] == "nsgan":
                    c_loss = nsgan_critic_train_loop(
                        reals=reals,
                        fakes=fakes,
                        critic=critic,
                        c_opt=c_opt,
                        c_scaler=c_scaler,
                        batch_idx=batch_idx,
                        autocast_enabled=is_mixed_precision,
                        grad_clip_val=training_config["grad_clip_val"],
                        r1_interval=training_config["strategy"]["params"][
                            "r1_interval"
                        ],
                        r1_coef=training_config["strategy"]["params"]["r1_coef"],
                    )
                mlflow.log_metric("c_loss_train_step", c_loss, step=cur_training_step)
                c_loss_mean += c_loss
                pbar.set_postfix(
                    {
                        "g_loss": g_loss,
                        "c_loss": c_loss,
                        training_config.get("monitor", "train_fid"): best_fid,
                    }
                )
                pbar.update()
            # ---------------------------------------------------------------------------- #
            #                          computing training metrics                          #
            # ---------------------------------------------------------------------------- #
            train_fid = fid_metric.compute().item()
            fid_metric.reset()
            mlflow.log_metric("train_fid_epoch", train_fid, step=epoch)
            mlflow.log_metric(
                "c_loss_train_epoch", c_loss_mean / (batch_idx + 1), step=epoch
            )
            mlflow.log_metric(
                "g_loss_train_epoch", g_loss_mean / (batch_idx + 1), step=epoch
            )
            pbar.close()

            # ---------------------------------------------------------------------------- #
            #                          validation section in epoch                         #
            # ---------------------------------------------------------------------------- #
            g_loss_val, c_loss_val = 0, 0
            batch_idx = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    reals, title_seq, z, noise = generate_inputs(
                        batch, device, generator.output_shape, generator.latent_dim
                    )
                    fakes = generator(z, title_seq, noise)
                    logits_f = critic(fakes)
                    logits_r = critic(reals)
                    g_loss_val += ns_gan_gen_loss(logits_f).item()
                    c_loss_val += ns_gan_disc_loss(logits_r, logits_f).item()
                    fid_metric.update(generator.to_uint8(reals), real=True)
                    fid_metric.update(generator.to_uint8(fakes), real=False)

            # ---------------------------------------------------------------------------- #
            #                         computing validation metrics                         #
            # ---------------------------------------------------------------------------- #
            val_fid = fid_metric.compute().item()
            fid_metric.reset()
            mlflow.log_metric("val_fid", val_fid, step=epoch)
            mlflow.log_metric(
                "c_loss_val_epoch", c_loss_val / (batch_idx + 1), step=epoch
            )
            mlflow.log_metric(
                "g_loss_val_epoch", g_loss_val / (batch_idx + 1), step=epoch
            )
            if training_config["monitor"] == "val-fid":
                condition = best_fid > val_fid
                if condition:
                    best_fid = val_fid
            else:
                condition = best_fid > train_fid
                if condition:
                    best_fid = train_fid
            if condition is True:
                save_model(
                    checkpoint_path,
                    generator,
                    g_opt,
                    g_scaler,
                    critic,
                    c_opt,
                    c_scaler,
                    epoch,
                )


if __name__ == "__main__":
    train()

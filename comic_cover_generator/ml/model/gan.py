"""Model module."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from comic_cover_generator.ml.dataset import CoverDataset
from comic_cover_generator.ml.loss import (
    critic_loss_fn,
    generator_loss_fn,
    gradient_penalty,
)
from comic_cover_generator.ml.model import Critic, Generator
from comic_cover_generator.ml.model.diffaugment import AugmentPolicy, diff_augment
from comic_cover_generator.ml.utils import CoverDatasetBatch, collate_fn
from comic_cover_generator.typing import TypedDict


class OptimizerParams(TypedDict):
    """Optimizer parameters."""

    cls: Type[torch.optim.Optimizer]
    kwargs: Dict[str, Any]


@dataclass
class ValidationData:
    """Validation data class."""

    z: torch.Tensor
    seq: Sequence[torch.Tensor]


# TODO: Make this class configurable.
class GAN(pl.LightningModule):
    """GAN class."""

    def __init__(
        self,
        optimizer_params: Dict[str, OptimizerParams] = None,
        batch_size: int = 1,
        gradient_penalty_coef: float = 0.2,
    ) -> None:
        """Instantiate a GAN object.

        Args:
            optimizer_params (Dict[str, OptimizerParams], optional): The optimizers parameters. Defaults to None.
            batch_size (int, optional): Defaults to 1.
            critic_update_step (int, optional): Defaults to 3.
            critic_loss_threshold (float, optional): Defaults to 0.3.
            gradient_penalty_coef (float, optional): Defaults to 0.2.
        """
        super().__init__()

        self.fid = FrechetInceptionDistance(feature=192, compute_on_step=False)

        self.train_dataset = None
        self.batch_size = batch_size
        self.gradient_penalty_coef = gradient_penalty_coef

        self.generator = Generator()
        self.critic = Critic()

        self.save_hyperparameters()

        if optimizer_params is None:
            default_params = {
                "cls": torch.optim.AdamW,
                "kwargs": {
                    "lr": 3e-4,
                    "weight_decay": 0.01,
                },
                "n_repeated_updates": 1,
            }
            optimizer_params = OrderedDict(
                {
                    "generator": default_params,
                    "critic": default_params,
                }
            )
        self.optimizer_params = optimizer_params

        self.validation_data: ValidationData = None

        self.augmentation_policy = [
            AugmentPolicy.color.value,
            AugmentPolicy.cutout.value,
            AugmentPolicy.translation.value,
        ]

    def attach_train_dataset_and_generate_validtaion_data(
        self, train_dataset: CoverDataset, val_gen_seed: int = 42
    ):
        """Attach train dataset and generate validation data.

        Args:
            train_dataset (Dataset): train dataset.
            val_gen_seed (int): validation generation seed.
        """
        self.train_dataset = train_dataset
        rng = np.random.default_rng(seed=val_gen_seed)
        seq = sorted(
            [
                self.train_dataset[idx]["title_seq"]
                for idx in rng.choice(
                    range(len(self.train_dataset)), size=8, replace=False
                )
            ],
            key=lambda x: x.size(0),
            reverse=True,
        )
        self.validation_data = ValidationData(
            torch.empty(8, Generator.latent_dim).normal_(mean=0.0, std=1.0), seq
        )

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure the optimizers of the model.

        Returns:
            List[torch.optim.Optimizer]:
        """
        return [
            {
                "optimizer": v["cls"](self.parameters(), **v["kwargs"]),
                "frequency": v["n_repeated_updates"],
            }
            for _, v in self.optimizer_params.items()
        ]

    def forward(self, z: torch.Tensor, seq: Sequence[torch.Tensor]) -> torch.Tensor:
        """Forward calls only the generator.

        Args:
            z (torch.Tensor): noise input.
            seq (Sequence[torch.Tensor]): sequence input.

        Returns:
            torch.Tensor: generated image.
        """
        return self.generator(z, seq)

    def training_step(
        self, batch: CoverDatasetBatch, batch_idx: int, optimizer_idx: int
    ) -> Dict[str, Any]:
        """Execute a training step.

        Args:
            batch (CoverDatasetBatch):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            Dict[str, Any]: The output of the training step.
        """

        def to_uint8(x: torch.Tensor):
            x = x / 2 + 1 / 2
            return (x * 255.0).type(torch.uint8)

        reals = batch["image"]
        seq = [x.to(reals.device) for x in batch["title_seq"]]

        # sorting seq and reals
        sorted_indices = (
            torch.tensor([s.size(0) for s in seq]).argsort(descending=True).tolist()
        )
        seq = [seq[idx] for idx in sorted_indices]
        reals = reals[sorted_indices, ...]

        # sample noise from normal distribution
        z = torch.empty(
            reals.size(0),
            self.generator.latent_dim,
            dtype=reals.dtype,
            device=reals.device,
        ).normal_(mean=0, std=1)

        # train generator
        if optimizer_idx == 0:

            self.critic.freeze()
            self.critic.eval()

            self.generator.unfreeze()
            self.generator.train()

            fake = self.generator(z, seq)
            critic_gen_fake = self.critic(fake).reshape(-1)
            loss_gen = generator_loss_fn(critic_gen_fake)
            tqdm_dict = {"generator_loss": loss_gen.detach()}
            output = {
                "loss": loss_gen,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
            self.log("generator_loss", tqdm_dict["generator_loss"], prog_bar=True)

            self.fid.update(to_uint8(fake.detach()), real=False)
            self.fid.update(to_uint8(reals), real=True)

            return output

        # train discriminator
        if optimizer_idx == 1:

            self.critic.unfreeze()
            self.critic.train()

            self.generator.freeze()
            self.generator.eval()

            fakes = self.generator(z, seq)

            fakes = diff_augment(fakes, self.augmentation_policy)
            reals = diff_augment(reals, self.augmentation_policy)

            critic_score_fakes = self.critic(fakes)
            critic_score_reals = self.critic(reals)
            gp = gradient_penalty(self.critic, reals.data, fakes.data)
            loss_critic = critic_loss_fn(
                critic_score_fakes, critic_score_reals, gp, self.gradient_penalty_coef
            )

            tqdm_dict = {"critic_loss": loss_critic.detach()}
            output = {
                "loss": loss_critic,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
            self.log(
                "critic_loss",
                tqdm_dict["critic_loss"],
                prog_bar=True,
            )
            return output

    def training_epoch_end(self, outputs: Any) -> None:
        """Generate an output on epoch end callback.

        Args:
            outputs (Any):

        Returns:
            None:
        """
        self.eval()

        w = next(filter(lambda x: x.requires_grad, self.parameters()))
        z = self.validation_data.z.type_as(w).to(w.device)
        seq = [s.to(w.device) for s in self.validation_data.seq]

        # log sampled images
        with torch.no_grad():
            sample_imgs = self(z, seq)

        grid = torchvision.utils.make_grid(
            sample_imgs, nrow=4, value_range=(-1, 1), normalize=True
        )

        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        # compute fid
        fid = self.fid.compute()
        self.log("fid", fid, prog_bar=True)
        self.fid.reset()

        return super().training_epoch_end(outputs)

    def train_dataloader(self) -> DataLoader:
        """Get the train loader.

        Returns:
            DataLoader: train data loader.
        """
        import psutil

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=psutil.cpu_count(logical=False),
        )

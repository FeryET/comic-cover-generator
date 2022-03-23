"""Model module."""
from collections import OrderedDict
from typing import Any, Dict, List, Type

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance

from comic_cover_generator.ml.loss import (
    critic_loss_fn,
    generator_loss_fn,
    gradient_penalty,
)
from comic_cover_generator.ml.model import Critic, Generator
from comic_cover_generator.typing import TypedDict


class OptimizerParams(TypedDict):
    """Optimizer parameters."""

    cls: Type[torch.optim.Optimizer]
    kwargs: Dict[str, Any]


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

        self.validation_z = torch.randn(16, self.generator.latent_dim)

    def attach_train_dataset(self, train_dataset: Dataset):
        """Attach train dataset.

        Args:
            train_dataset (Dataset): train dataset.
        """
        self.train_dataset = train_dataset

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward calls only the generator.

        Args:
            z (torch.Tensor): noise input.

        Returns:
            torch.Tensor: generated image.
        """
        return self.generator(z)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, optimizer_idx: int
    ) -> Dict[str, Any]:
        """Execute a training step.

        Args:
            batch (Dict[str, torch.Tensor]):
            batch_idx (int):
            optimizer_idx (int):

        Returns:
            Dict[str, Any]: The output of the training step.
        """

        def to_uint8(x: torch.Tensor):
            return (x * 255.0).type(torch.uint8)

        reals: torch.Tensor = batch["image"]

        device = reals.device

        # sample noise
        z = torch.randn(
            reals.shape[0], self.generator.latent_dim, dtype=reals.dtype, device=device
        )

        # train generator
        if optimizer_idx == 0:

            self.critic.freeze()
            self.critic.eval()

            self.generator.unfreeze()
            self.generator.train()

            fake = self.generator(z)
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

            fakes = self.generator(z)
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
        z = self.validation_z.type_as(w).to(w.device)

        # log sampled images
        with torch.no_grad():
            sample_imgs = self(z)

        grid = torchvision.utils.make_grid(sample_imgs, nrow=4)

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

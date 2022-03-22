"""Model module."""
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset

from comic_cover_generator.ml.loss import (
    critic_loss_fn,
    generator_loss_fn,
    gradient_penalty,
)
from comic_cover_generator.typing import Protocol, TypedDict


class Freezeable(Protocol):
    """Protocol for freezable nn.Modules."""

    def freeze(self):
        """Freezing method."""
        ...

    def unfreeze(self):
        """Unfreezing method."""
        ...


class Critic(nn.Module, Freezeable):
    """critic Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (128, 128)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 4, 2, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.Conv2d(32, 128, 3, 4, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.AdaptiveMaxPool2d((2, 2)),
        )
        self.clf = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=2, padding=0, stride=2), nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.clf(self.features(x))

    def freeze(self):
        """Freeze the critic model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the critic model."""
        for p in self.parameters():
            p.requires_grad = True
        return self


class Generator(nn.Module, Freezeable):
    """Generator model based on PGAN."""

    latent_dim: int = 512
    output_shape: Tuple[int, int] = (256, 256)

    def __init__(self, pretrained=True) -> None:
        """Initialize an instance.

        Args:
            pretrained (bool, optional): Defaults to True.
            use_apu (bool, optional): Defaults to False.
        """
        super().__init__()
        self.features = torch.hub.load(
            "facebookresearch/pytorch_GAN_zoo:hub",
            "PGAN",
            model_name="celebAHQ-256",
            pretrained=pretrained,
        ).getNetG()
        self.resizer = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.resizer(self.features(x))

    def freeze(self):
        """Freeze the generator model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the generator model."""
        for p in self.parameters():
            p.requires_grad = True
        return self


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
        pretrained: bool = True,
        batch_size: int = 1,
        gradient_penalty_coef: float = 0.2,
    ) -> None:
        """Instantiate a GAN object.

        Args:
            optimizer_params (Dict[str, OptimizerParams], optional): The optimizers parameters. Defaults to None.
            pretrained (bool, optional): Defaults to True.
            batch_size (int, optional): Defaults to 1.
            critic_update_step (int, optional): Defaults to 3.
            critic_loss_threshold (float, optional): Defaults to 0.3.
            gradient_penalty_coef (float, optional): Defaults to 0.2.
        """
        super().__init__()

        self.train_dataset = None
        self.batch_size = batch_size
        self.gradient_penalty_coef = gradient_penalty_coef

        self.generator = Generator(pretrained=pretrained)
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

        self.validation_z = torch.randn(8, self.generator.latent_dim)

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
        reals: torch.Tensor = batch["image"]

        device = reals.device

        # sample noise
        z = torch.randn(
            reals.shape[0], self.generator.latent_dim, dtype=reals.dtype, device=device
        )

        # train generator
        if optimizer_idx == 0:
            self.critic.freeze()
            self.generator.unfreeze()
            gen_fake = self.critic(self.generator(z)).reshape(-1)
            loss_gen = generator_loss_fn(gen_fake)
            tqdm_dict = {"generator_loss": loss_gen.detach()}
            output = {
                "loss": loss_gen,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
            self.log("generator_loss", tqdm_dict["generator_loss"], prog_bar=True)
            return output

        # train discriminator
        if optimizer_idx == 1:
            self.critic.unfreeze()
            self.generator.freeze()
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

    def on_epoch_end(self):
        """Generate an output on epoch end callback."""
        z = self.validation_z.type_as(self.generator.resizer.weight).to(
            self.generator.resizer.weight.device
        )

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

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

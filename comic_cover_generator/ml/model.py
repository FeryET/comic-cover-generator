"""Model module."""
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from comic_cover_generator.typing import Protocol, TypedDict


class Freezeable(Protocol):
    """Protocol for freezable nn.Modules."""

    def freeze(self):
        """Freezing method."""
        ...

    def unfreeze(self):
        """Unfreezing method."""
        ...


class Discriminator(nn.Module, Freezeable):
    """Discriminator Model based on MobileNetV3."""

    input_shape: Tuple[int, int] = (224, 224)

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize an instance.

        Args:
            pretrained (bool, optional): Defaults to True.
        """
        super().__init__()
        self.features = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        self.clf = nn.Linear(1000, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        return self.clf(self.features(x))

    def freeze(self):
        """Freeze the discriminator model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the discriminator model."""
        for p in self.parameters():
            p.requires_grad = True
        return self


class Generator(nn.Module, Freezeable):
    """Generator model based on PGAN."""

    latent_dim: int = 512
    output_shape: Tuple[int, int] = (224, 224)

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

        # resize 256x256 to 224x224
        self.resizer = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=33,
            stride=1,
            padding=0,
            groups=3,
        )

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
    ) -> None:
        """Instantiate a GAN object.

        Args:
            optimizer_params (Dict[str, OptimizerParams], optional): The optimizers parameters. Defaults to None.
            pretrained (bool, optional): Defaults to True.
            batch_size (in, optional): Defaults to 1.
        """
        super().__init__()

        self.batch_size = batch_size

        self.generator = Generator(pretrained=pretrained)
        self.discriminator = Discriminator(pretrained=pretrained)

        self.save_hyperparameters()

        if optimizer_params is None:
            default_params = {
                "cls": torch.optim.AdamW,
                "kwargs": {
                    "lr": 3e-4,
                    "weight_decay": 0.01,
                },
            }
            optimizer_params = OrderedDict(
                {
                    "generator": default_params,
                    "discriminator": default_params,
                }
            )
        self.optimizer_params = optimizer_params

        self.adversarial_loss = nn.BCEWithLogitsLoss()

        self.validation_z = torch.randn(8, self.generator.latent_dim)

    def make_partially_trainable(self):
        """Make some layers partially trainable in the model."""
        self.generator.unfreeze()
        self.discriminator.unfreeze()

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure the optimizers of the model.

        Returns:
            List[torch.optim.Optimizer]:
        """
        return [
            v["cls"](self.parameters(), **v["kwargs"])
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
        imgs: torch.Tensor = batch["image"]

        device = imgs.device

        # sample noise
        z = torch.randn(
            imgs.shape[0], self.generator.latent_dim, dtype=imgs.dtype, device=device
        )

        # train generator
        if optimizer_idx == 0:

            self.generator.unfreeze()
            self.discriminator.freeze()

            valid = torch.ones(imgs.size(0), 1, device=device)
            valid = valid.type_as(imgs)
            generator_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"generator_loss": generator_loss.detach()}
            output = {
                "loss": generator_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
            self.log("generator_loss", tqdm_dict["generator_loss"], prog_bar=True)
            return output

        # train discriminator
        if optimizer_idx == 1:

            self.generator.freeze()
            self.discriminator.unfreeze()

            valid = torch.ones(imgs.size(0), 1, dtype=imgs.dtype, device=device)
            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)
            fake = torch.zeros(imgs.size(0), 1, dtype=imgs.dtype, device=device)
            fake_loss = self.adversarial_loss(self.discriminator(self(z)), fake)
            discrimantor_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"discriminator_loss": discrimantor_loss.detach()}
            output = {
                "loss": discrimantor_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
            self.log(
                "discriminator_loss", tqdm_dict["discriminator_loss"], prog_bar=True
            )
            return output

    def on_epoch_end(self):
        """Generate an output on epoch end callback."""
        z = self.validation_z.type_as(self.generator.resizer.weight).to(
            self.generator.resizer.weight.device
        )

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

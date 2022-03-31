"""Model module."""
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple, Type

import pytorch_lightning as pl
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.model import Critic, Generator
from comic_cover_generator.ml.model.critic import CriticParams
from comic_cover_generator.ml.model.diffaugment import AugmentPolicy
from comic_cover_generator.ml.model.generator import GeneratorParams
from comic_cover_generator.ml.model.training_strategy.base import (  # noqa: sort
    TrainingStrategy,
)
from comic_cover_generator.ml.utils import CoverDatasetBatch
from comic_cover_generator.typing import TypedDict


class TrainingStrategyParams(TypedDict):
    """TrainingStrategy Parameters."""

    cls: Type[TrainingStrategy]
    kwargs: Dict[str, Any]


class OptimizerParams(TypedDict):
    """Optimizer parameters."""

    cls: Type[torch.optim.Optimizer]
    kwargs: Dict[str, Any]


# TODO: Make this class configurable.
class GAN(pl.LightningModule):
    """GAN class."""

    def __init__(
        self,
        training_strategy_params: TrainingStrategyParams,
        generator_params: GeneratorParams = None,
        critic_params: CriticParams = None,
        optimizer_params: Dict[str, OptimizerParams] = None,
    ) -> None:
        """Instantiate a GAN object.

        Args:
            generator_params: (GeneratorParams, optional): Defaults to None.
            critic_params (CriticParams, optional): Defaults to None.
            optimizer_params (Dict[str, OptimizerParams], optional): The optimizers parameters. Defaults to None.

        Returns:
            None:
        """
        super().__init__()

        self.train_dataset = None

        self.training_strategy: TrainingStrategy = training_strategy_params["cls"](
            **training_strategy_params["kwargs"]
        )
        self.training_strategy.attach_model(self)

        self.save_hyperparameters()

        if optimizer_params is None:
            default_params = {
                "cls": torch.optim.AdamW,
                "kwargs": {
                    "lr": 3e-4,
                    "weight_decay": 0.01,
                    "betas": (0.0, 0.9),
                    "eps": Constants.eps,
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

        if generator_params is None:
            generator_params = {}

        self.generator = Generator(**generator_params)

        if critic_params is None:
            critic_params = {}

        self.critic = Critic(**critic_params)

        self.augmentation_policy = [
            AugmentPolicy.color.value,
            AugmentPolicy.cutout.value,
            AugmentPolicy.translation.value,
        ]

        self.train_fid = FrechetInceptionDistance(feature=192, compute_on_step=False)
        self.val_fid = self.train_fid.clone()

        self.G_OPT_INDEX, self.C_OPT_INDEX = None, None

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizers of the model.

        Returns:
            Dict[str, Any]:
        """
        # address epsilon values
        for key in self.optimizer_params:
            cls, kwargs = (
                self.optimizer_params[key]["cls"],
                self.optimizer_params[key]["kwargs"],
            )
            if "eps" not in kwargs and "eps" in cls.__init__.__code__.co_varnames:
                kwargs["eps"] = Constants.eps

        # generator optimizer
        gen_params = self.optimizer_params["generator"]
        gen_lr = gen_params["kwargs"]["lr"]
        g_opt = gen_params["cls"](
            self.generator.get_optimizer_parameters(gen_lr), **gen_params["kwargs"]
        )
        # critic optimizer
        critic_params = self.optimizer_params["critic"]
        c_opt = critic_params["cls"](
            self.critic.parameters(), **critic_params["kwargs"]
        )
        from comic_cover_generator.ml.model.training_strategy.wgan_gp import (
            WGANPlusGPTrainingStrategy,
        )

        if isinstance(self.training_strategy, WGANPlusGPTrainingStrategy):
            g_frequency = 1
            c_frequency = self.training_strategy.critic_update_interval
        else:
            g_frequency = 1
            c_frequency = 1

        self.G_OPT_INDEX, self.C_OPT_INDEX = 0, 1

        return {
            "optimizer": g_opt,
            "frequency": g_frequency,
        }, {"optimizer": c_opt, "frequency": c_frequency}

    def forward(
        self,
        z: torch.Tensor,
        seq: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Forward calls only the generator.

        Args:
            z (torch.Tensor): Latent noise input.
            seq (Sequence[torch.Tensor]): sequence input.

        Returns:
            torch.Tensor: generated image.
        """
        return self.generator(z, seq)

    def _extract_inputs(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:

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

        return reals, z, seq

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
        reals, z, seq = self._extract_inputs(batch)
        B = reals.size(0)
        # train generator
        if optimizer_idx == 0:
            self.critic.freeze()
            self.critic.eval()
            self.generator.unfreeze()
            self.generator.train()

            output = self.training_strategy.generator_loop(seq, z, batch_idx)
            fakes = output["fakes"]

            self.train_fid.update(self.generator.to_uint8(fakes.detach()), real=False)
            self.train_fid.update(self.generator.to_uint8(reals), real=True)
            k = "generator"

        # train discriminator
        if optimizer_idx == 1:
            self.critic.unfreeze()
            self.critic.train()
            self.generator.freeze()
            self.generator.eval()

            output = self.training_strategy.critic_loop(reals, seq, z, batch_idx)
            k = "critic"

        self.log(
            f"train_{k}_loss",
            output["loss"].detach(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=B,
        )
        return output

    def training_epoch_end(self, outputs: Any) -> None:
        """Generate an output on epoch end callback.

        Args:
            outputs (Any):

        Returns:
            None:
        """
        # compute fid
        fid = self.train_fid.compute()
        self.log("train_fid", fid, prog_bar=True)
        self.train_fid.reset()

        return super().training_epoch_end(outputs)

    def validation_step(
        self, batch: CoverDatasetBatch, batch_idx: int, *args, **kwargs
    ) -> Dict[str, Any]:
        """Apply a validation step.

        Args:
            batch (CoverDatasetBatch):
            batch_idx (int):

        Returns:
            Dict[str, Any]:
        """
        reals, z, seq = self._extract_inputs(batch)
        B = reals.size(0)

        output = self.training_strategy.validation_loop(reals, seq, z, batch_idx)

        fakes = output["fakes"]

        self.val_fid.update(self.generator.to_uint8(fakes), real=False)
        self.val_fid.update(self.generator.to_uint8(reals), real=True)

        self.log_dict(
            {
                "val_generator_loss": output["generator_loss"],
                "val_critic_loss": output["critic_loss"],
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=B,
        )

        if batch_idx == 0:
            if self.current_epoch == 0:
                self.validation_z = z[:16]
                self.validation_seq = seq[:16]
                sample_imgs = fakes[:16]
            else:
                sample_imgs = self(self.validation_z, self.validation_seq)
            # log sampled images
            grid = torchvision.utils.make_grid(
                sample_imgs, nrow=4, value_range=(-1, 1), normalize=True
            )

            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )

        return {"loss": output["critic_loss"]}

    def validation_epoch_end(self, outputs: Any) -> None:
        """Update metrics on the validation end.

        Args:
            outputs (Any):

        Returns:
            None:
        """
        fid = self.val_fid.compute()
        self.log("val_fid", fid, prog_bar=True)
        self.val_fid.reset()
        return super().validation_epoch_end(outputs)

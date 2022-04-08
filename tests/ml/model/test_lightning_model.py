import random
from unittest import mock

import pytest
import pytorch_lightning
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance

from comic_cover_generator.ml.model import GAN, Critic, Generator
from comic_cover_generator.ml.model.diffaugment import diff_augment
from comic_cover_generator.ml.model.training_strategy.wgan_gp import (
    WGANPlusGPTrainingStrategy,
)


@pytest.fixture(scope="function")
def model():
    with mock.patch(
        "comic_cover_generator.ml.model.gan.Generator", autospec=Generator
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.Critic", autospec=Critic
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.FrechetInceptionDistance",
        spec=FrechetInceptionDistance,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.torch", autospec=torch
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.pl",
        autospec=pytorch_lightning,
    ):
        gan = GAN(
            training_strategy_params={
                "cls": mock.MagicMock(autospec=WGANPlusGPTrainingStrategy),
                "kwargs": {},
            }
        )
        yield gan


@pytest.fixture(scope="module", params=[1, 5, 10])
def batch(request):
    with mock.patch(
        "comic_cover_generator.ml.model.gan.torch.Tensor", spec=torch.Tensor
    ), mock.patch.object(torch, "rand", spec=torch.rand), mock.patch.object(
        torch, "randint", spec=torch.randint
    ):
        return {
            "image": torch.rand(request.param, 3, 64, 64),
            "title_seq": [
                torch.randint(0, 256, [random.randint(1, 3)])
                for _ in range(request.param)
            ],
        }


@torch.no_grad()
@pytest.mark.parametrize("optimizer_idx", argvalues=[0, 1])
def test_training_step_input_pass(model, batch, optimizer_idx):
    with mock.patch.object(
        model,
        "_extract_inputs",
        return_value=(torch.ones(1), torch.zeros(1), torch.rand(1), torch.rand(1)),
    ), mock.patch.object(model, "log", autospec=True):
        model.training_step(batch, 0, optimizer_idx)
        model._extract_inputs.assert_called_once()
        if optimizer_idx == 0:
            model.generator.unfreeze.assert_called_once()
            model.critic.freeze.assert_called_once()
            model.training_strategy.generator_loop.assert_called_once()
        if optimizer_idx == 1:
            model.generator.freeze.assert_called_once()
            model.critic.unfreeze.assert_called_once()
            model.training_strategy.critic_loop.assert_called_once()
        model.log.assert_called_once()

import random
from unittest import mock

import pytest
import pytorch_lightning
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance

from comic_cover_generator.ml.model import GAN, Critic, Generator
from comic_cover_generator.ml.model.diffaugment import diff_augment


def append_freeze_unfreeeze(mocked):
    mocked.freeze = mock.MagicMock()
    mocked.unfreeze = mock.MagicMock()
    return mocked


@pytest.fixture(scope="function")
def model():
    result_func = lambda *args, **kwargs: torch.Tensor([1.0])  # noqa:
    with mock.patch(
        "comic_cover_generator.ml.model.gan.Generator", autospec=Generator
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.Critic", autospec=Critic
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.gradient_penalty",
        result_func,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.critic_loss_fn",
        result_func,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.generator_loss_fn",
        result_func,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.FrechetInceptionDistance",
        spec=FrechetInceptionDistance,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.torch", autospec=torch
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.pl",
        autospec=pytorch_lightning,
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.diff_augment", spec=diff_augment
    ):
        gan = GAN()
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
def test_training_step_input_pass(model: GAN, batch, optimizer_idx):
    with mock.patch.object(
        model, "_training_step_generator", return_value={"loss": torch.Tensor([1])}
    ), mock.patch.object(
        model,
        "_training_step_critic",
        return_value={"loss": torch.Tensor([1])},
    ), mock.patch.object(
        model, "_extract_inputs", return_value=(1, 2, 3)
    ):
        model.training_step(batch, 0, optimizer_idx)
        model._extract_inputs.assert_called_once()
        if optimizer_idx == 0:
            model._training_step_generator.assert_called_once()
        if optimizer_idx == 1:
            model._training_step_critic.assert_called_once()

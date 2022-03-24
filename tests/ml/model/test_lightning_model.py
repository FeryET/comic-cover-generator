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


@pytest.fixture(scope="module")
def model():
    result_func = lambda *args, **kwargs: torch.Tensor([1.0])  # noqa:
    with mock.patch(
        "comic_cover_generator.ml.model.gan.Generator", spec=Generator
    ), mock.patch("comic_cover_generator.ml.model.gan.Critic", spec=Critic), mock.patch(
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
        "comic_cover_generator.ml.model.gan.torch", spec=torch
    ), mock.patch(
        "comic_cover_generator.ml.model.gan.pl",
        spec=pytorch_lightning,
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
            "image": torch.rand(request.param, 3, *Critic.input_shape),
            "title_seq": [
                torch.randint(0, 256, [random.randint(1, 3)])
                for i in range(request.param)
            ],
        }


@torch.no_grad()
@pytest.mark.parametrize("optimizer_idx", argvalues=[0, 1])
def test_training_step_input_pass(model, batch, optimizer_idx):
    model.training_step(batch, 0, optimizer_idx)

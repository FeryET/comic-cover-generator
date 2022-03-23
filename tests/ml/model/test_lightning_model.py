from unittest import mock

import pytest
import torch
from torch import nn

from comic_cover_generator.ml.model import GAN, Critic, Generator


def append_freeze_unfreeeze(mocked):
    mocked.freeze = mock.MagicMock()
    mocked.unfreeze = mock.MagicMock()
    return mocked


@pytest.fixture(scope="module")
def model():
    result_func = lambda *args, **kwargs: torch.Tensor([1.0])  # noqa:
    with mock.patch("comic_cover_generator.ml.model.gan.Generator"), mock.patch(
        "comic_cover_generator.ml.model.gan.Critic"
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
    ), mock.patch(
        "torch.Tensor.backward"
    ):
        gan = GAN()
        yield gan


@pytest.fixture(scope="module", params=[1, 5, 10])
def batch(request):
    return {"image": torch.rand(request.param, 3, *Critic.input_shape)}


@torch.no_grad()
@pytest.mark.parametrize("optimizer_idx", argvalues=[0, 1])
def test_training_step_input_pass(model, batch, optimizer_idx):
    model.training_step(batch, 0, optimizer_idx)

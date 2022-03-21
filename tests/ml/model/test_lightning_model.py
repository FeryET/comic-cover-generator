from unittest import mock

import pytest
import torch
from torch import nn

from comic_cover_generator.ml.model import GAN, Discriminator, Generator


def append_freeze_unfreeeze(mocked):
    mocked.freeze = mock.MagicMock()
    mocked.unfreeze = mock.MagicMock()
    return mocked


@pytest.fixture(scope="module")
def model():
    with mock.patch("comic_cover_generator.ml.model.Generator") as gen_mock:
        with mock.patch("comic_cover_generator.ml.model.Discriminator") as disc_mock:
            gan = GAN(pretrained=False)
            gan.fake_loss_fn = mock.MagicMock(return_value=torch.ones(1, 1))
            gan.real_loss_fn = mock.MagicMock(return_value=torch.ones(1, 1))
            gan.gradient_penalty_fn = mock.MagicMock(return_value=torch.rand(1))
            yield gan


@pytest.fixture(scope="function")
def reset_model_state(model):
    for m in [model.fake_loss_fn, model.real_loss_fn, model.gradient_penalty_fn]:
        m.reset_mock()


@pytest.fixture(scope="module", params=[1, 5, 10])
def batch(request):
    return {"image": torch.rand(request.param, *Discriminator.input_shape)}


@torch.no_grad()
@mock.patch("torch.Tensor.backward", lambda: None)
@pytest.mark.parametrize("optimizer_idx", argvalues=[0, 1])
def test_training_step_input_pass(model, batch, optimizer_idx, reset_model_state):
    model.training_step(batch, 0, optimizer_idx)

    if optimizer_idx == 0:
        model.real_loss_fn.assert_called_once()
        model.fake_loss_fn.assert_not_called()
        model.gradient_penalty_fn.assert_not_called()
    else:
        model.real_loss_fn.assert_called_once()
        model.fake_loss_fn.assert_called_once()
        model.gradient_penalty_fn.assert_called_once()

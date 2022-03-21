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
            gen_mock.return_value = mock.MagicMock(spec=Generator)
            disc_mock.return_value = mock.MagicMock(spec=Discriminator)
            gan = GAN(pretrained=False)
            gan.fake_loss_fn = gan.real_loss_fn = lambda x: torch.ones(1, 1)
            yield gan


@pytest.fixture(scope="module", params=[1, 5, 10])
def batch(request):
    return {"image": torch.rand(request.param, *Discriminator.input_shape)}


def test_make_partially_trainable_correct_call(model: GAN):
    model.make_partially_trainable()
    model.generator.unfreeze.assert_called_once()
    model.discriminator.unfreeze.assert_called_once()


@torch.no_grad()
@mock.patch("torch.Tensor.backward", lambda: None)
@pytest.mark.parametrize("optimizer_idx", argvalues=[0, 1])
def test_training_step_input_pass(model, batch, optimizer_idx):
    model.generator.reset_mock()
    model.discriminator.reset_mock()

    model.training_step(batch, 0, optimizer_idx)

    if optimizer_idx == 0:
        model.generator.unfreeze.assert_called_once()
        model.discriminator.freeze.assert_called_once()
    else:
        model.generator.freeze.assert_called_once()
        model.discriminator.unfreeze.assert_called_once()

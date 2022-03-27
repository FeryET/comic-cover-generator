import pytest
import torch

from tests.fixtures.constants import image_size

from comic_cover_generator.ml.model.base import ModulatedConv2D


@pytest.mark.parametrize(
    "in_channels, out_channels, eps",
    argvalues=([1, 1, 0.001], [2, 1, 1e-10], [10, 10, 1e-2]),
)
def test_modulated_conv2d_intialization_pass(in_channels, out_channels, eps):
    ModulatedConv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        eps=eps,
    )


@pytest.mark.parametrize(
    "out_channels, input_shape",
    argvalues=(
        [7, (2, 9, 10, 10)],
        [5, (3, 1, 30, 30)],
        [10, (4, 2, 20, 20)],
        [4, (5, 3, 10, 10)],
    ),
)
def test_modulated_conv2d_forward_pass(out_channels, input_shape):
    B, C, H, W = input_shape
    conv2d = ModulatedConv2D(
        in_channels=C,
        out_channels=out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
    )
    conv2d.forward(
        torch.rand(B, C, H, W),
        torch.rand(B, C),
    )

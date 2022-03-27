"""Generator module."""
import math
from collections import namedtuple
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from comic_cover_generator.ml.model.base import Freezeable, ModulatedConv2D, Seq2Vec
from comic_cover_generator.ml.model.utils import weights_init


class Generator(nn.Module, Freezeable):
    """Generator model based on PGAN."""

    latent_dim: int = 256
    w_dim: int = 256
    sequence_embed_dim: int = 128
    output_shape: Tuple[int, int] = (64, 64)

    def __init__(self) -> None:
        """Initialize an instance."""
        super().__init__()

        self.register_parameter(
            "cte",
            nn.parameter.Parameter(
                torch.empty(1, 1024, 4, 4).normal_(mean=0, std=1), requires_grad=True
            ),
        )

        self.title_embed = nn.Sequential(
            Seq2Vec(self.sequence_embed_dim),
            nn.Linear(self.sequence_embed_dim, self.w_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.latent_mapper = LatentMapper(self.latent_dim, self.w_dim)

        channels = [
            # 4x4
            1024,
            # 8x8
            512,
            # 16x16
            256,
            # 32x32
            128,
            3,
            # 64x64
        ]
        channels = list(zip(channels, channels[1:]))

        img_dims = [4, 8, 16, 32, 64]
        img_dims = list(zip(img_dims, img_dims[1:]))

        self.features = nn.ModuleList()
        for (in_ch, out_ch), (in_dim, out_dim) in zip(channels, img_dims):
            self.features.append(
                GeneratorBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    w_dim=self.w_dim,
                    input_shape=(in_dim, in_dim),
                    output_shape=(out_dim, out_dim),
                )
            )

        self.apply(weights_init)

    def forward(
        self, z: torch.Tensor, title_seq: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        """Map a noise and a sequence to an image.

        Args:
            z (torch.Tensor): Latent noise input.
            title_seq (torch.Tensor): Title sequence.
            n (torch.Tensor): Synthesis noise input. Either 4D with shape [B, 1, H, W] or 3D with [B, H, W].

        Returns:
            torch.Tensor:
        """
        if n.ndim == 3:
            n = n.unsqueeze(1)
        if n.size()[2:] != self.output_shape:
            raise RuntimeError("Mismatch between generated noise and output shape.")

        B = z.size(0)
        # enriching the latent vector
        embed = self.title_embed(title_seq)
        mapped = self.latent_mapper(F.normalize(z, dim=-1))
        w = mapped + embed
        # upscaling the parameter
        x = self.cte.repeat(B, 1, 1, 1)
        rgb = None
        for f in self.features:
            x, rgb = f(x, w, n, rgb)
        return rgb

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


class LatentMapper(nn.Module):
    """A latent mapping module to map a latent vector to w plane."""

    def __init__(self, latent_dim: int, w_dim: int) -> None:
        """Initialize the mapper.

        Args:
            latent_dim (int): noise dimension.
            w_dim (int): mapped dimension.
        """
        super().__init__()

        self.mapper = nn.Sequential(
            nn.Unflatten(1, (1, latent_dim)),
            nn.Conv1d(1, w_dim // 4, 5, 2, 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(w_dim // 4, w_dim // 2, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(w_dim // 2, w_dim, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map x to w plane.

        Args:
            x (torch.Tensor): Input 2D tensor.

        Returns:
            torch.Tensor: Mapped 2D tensor.
        """
        return self.mapper(x)


class GeneratorBlock(nn.Module):
    """Generator block module."""

    GeneratorResult = namedtuple("GeneratorResult", "x,rgb")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
    ) -> None:
        """Intiailizes a generator block.

        Args:
            in_channels (int):
            out_channels (int):
            w_dim (int):
            input_shape (Tuple[int, int]):
            output_shape (Tuple[int, int]):

        Returns:
            None:
        """
        super().__init__()
        self.style1 = StyleBlock(
            w_dim,
            output_shape=input_shape,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.style2 = StyleBlock(
            w_dim,
            output_shape=output_shape,
            upsample=True,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.to_rgb = nn.Conv2d(out_channels, 3, 1, 1, 0)

        self.register_buffer("residual_scaler", torch.as_tensor([math.sqrt(0.5)]))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        n: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> GeneratorResult:
        """Forward call of generator block.

        Args:
            x (torch.Tensor): input 4D tensor.
            w (torch.Tensor): input 2D latent tensor.
            n (torch.Tensor): input 2D noise tensor.
            rgb (torch.Tensor, optional): residual input. Defaults to None.

        Returns:
            GeneratorResult:
        """
        x = self.style1(x, w, n)
        x = self.style2(x, w, n)
        if rgb is not None:
            rgb = self.to_rgb(x) + self.upsample(rgb)
            rgb = rgb * self.residual_scaler
        else:
            rgb = self.to_rgb(x)
        return GeneratorBlock.GeneratorResult(x=x, rgb=rgb)


def center_crop(noise: torch.Tensor, crop_shape: torch.Tensor) -> torch.Tensor:
    """Apply a center crop transformation on an input.

    Args:
        noise (torch.Tensor):
        crop_shape (torch.Tensor):

    Returns:
        torch.Tensor:
    """
    diff = torch.as_tensor(noise.size()[2:], device=noise.device) - crop_shape
    crop_start = diff.div(2, rounding_mode="floor")
    crop_end = crop_start + crop_shape
    return noise[
        ...,
        crop_start[0] : crop_end[0],
        crop_start[1] : crop_end[1],
    ]


class AdditiveNoiseBlock(nn.Module):
    """StyleGAN additive noise block."""

    def __init__(self, output_shape: Tuple[int, int]) -> None:
        """Initialize the additive noise module.

        Args:
            output_shape (Tuple[int, int]): The output shape of the noise block.
        """
        super().__init__()
        self.register_parameter(
            "coef", nn.parameter.Parameter(torch.rand(1), requires_grad=True)
        )
        self.register_buffer(
            "output_shape", torch.as_tensor(output_shape, dtype=torch.long)
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Compute a weighted view of the center cropped noise.

        Args:
            noise (torch.Tensor): 4D noise input. [B, 1, H, W]

        Returns:
            torch.Tensor: Cropped noise.
        """
        noise = center_crop(noise, self.output_shape)
        noise = noise * self.coef
        return noise


class StyleBlock(nn.Module):
    """Style block module."""

    def __init__(
        self,
        w_dim: int,
        output_shape: Tuple[int, int],
        eps: float = 1e-5,
        upsample: bool = False,
        **conv_params
    ) -> None:
        """Intiailize a generator block module.

        Args:
            w_dim (int): Fine grained latent dimension.
            output_shape (Tuple[int, int]): The dimensions of the output of the block.
            eps (float, optional): Defaults to 1e-5.
            upsample (bool, optional): Whether to upsample the input. Defaults to False.
        """
        super().__init__()

        self.mod_conv = ModulatedConv2D(eps=eps, **conv_params)
        self.B = AdditiveNoiseBlock(output_shape)
        self.A = nn.Linear(w_dim, self.mod_conv.conv.in_channels)

        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear") if upsample else None
        )

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        """Apply a modulated block for the generator.

        Args:
            x (torch.Tensor): Input 4D tensor.
            w (torch.Tensor): Input fine-grained latent 2D tensor.
            n (torch.Tensor): Input noise tensor.

        Returns:
            torch.Tensor: torch.Tensor
        """
        if self.upsample is not None:
            x = self.upsample(x)

        scores = self.A(w)
        # add conv and add bias
        x = self.mod_conv(x, scores)
        # add noise
        x = x + self.B(n)

        return self.activation(x)

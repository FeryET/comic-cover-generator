"""Generator module."""
import math
from collections import namedtuple
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchEncoding

from comic_cover_generator.ml.model.base import EqualLinear, Freezeable, ModulatedConv2D
from comic_cover_generator.ml.model.seq2vec import Seq2Vec
from comic_cover_generator.typing import TypedDict


class GeneratorParams(TypedDict):
    """Generator initialization arguments."""

    latent_dim: int
    w_dim: int
    conv_channels: Tuple[int, ...]
    output_shape: Tuple[int, int]
    n_heads: int
    dim_feedforward: int
    num_layers: int
    max_len: int
    padding_idx: int
    mapping_network_lr_coef: float


class Generator(nn.Module, Freezeable):
    """Generator model based on PGAN."""

    def __init__(
        self,
        latent_dim: int = 256,
        w_dim: int = 256,
        conv_channels: Tuple[int, ...] = (512, 256, 128, 64, 32),
        output_shape: Tuple[int, int] = (64, 64),
        mapping_network_lr_coef: float = 1e-2,
        transformer_model: str = "prajjwal1/bert-tiny",
    ) -> None:
        """Initialize the generator.

        Args:
            latent_dim (int, optional): Defaults to 256.
            w_dim (int, optional): Defaults to 256.
            conv_channels (Tuple[int, ...], optional): Defaults to (512, 256, 128, 64, 32).
            output_shape (Tuple[int, int], optional): Defaults to (64, 64).
            mapping_network_lr_coef (float, optional): Defaults to 1e-2.
            transformer_model (str): the transfomer model used in this generator.

        Raises:
            ValueError:
        """
        super().__init__()

        if output_shape[0] != int(4 * 2 ** (len(conv_channels) - 1)):
            computed_shape = tuple(
                int(4 * 2 ** (len(conv_channels) - 1)) for _ in range(2)
            )
            raise ValueError(
                f"Mismatch between output shape: {output_shape} and channels specified"
                f" which results in computed output shape of {computed_shape}. Each"
                " value in channels corresponds to a block which includes an 2x"
                " upsampler."
            )

        # mapping network properties
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        # output shape
        self.output_shape = output_shape
        self.mapping_network_lr_coef = mapping_network_lr_coef

        self.conv_channels = list(conv_channels)

        self.seq2vec = nn.Sequential(
            Seq2Vec(
                transformer_model=transformer_model,
                output_dim=self.conv_channels[0] * 4 * 4,
            ),
            nn.Unflatten(dim=-1, unflattened_size=(self.conv_channels[0], 4, 4)),
        )

        self.latent_mapper = LatentMapper(self.latent_dim, self.w_dim)

        self.features = nn.ModuleList([])
        for in_ch, out_ch in zip(self.conv_channels, self.conv_channels[1:]):
            self.features.append(
                GeneratorBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    w_dim=self.w_dim,
                    upsample=True,
                ),
            )

        self.rescale_rgb = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        title_seq: BatchEncoding,
    ) -> torch.Tensor:
        """Map a noise and a sequence to an image.

        Args:
            z (torch.Tensor): Latent noise input.
            title_seq (torch.Tensor): Title sequence.

        Returns:
            torch.Tensor:
        """
        # enriching the latent vector
        w = self.latent_mapper(z)
        # embed the sequence to a vector
        x = self.seq2vec(title_seq)
        rgb = None
        for f in self.features:
            x, rgb = f(x, w, rgb)
        rgb = self.rescale_rgb(rgb)
        return rgb

    def freeze(self):
        """Freeze the generator model."""
        for p in self.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self):
        """Unfreeze the generator model."""
        for p in nn.ModuleList([self.latent_mapper, self.features]).parameters():
            p.requires_grad = True
        self.seq2vec[0].unfreeze()
        return self

    def get_optimizer_parameters(self, lr: float) -> Tuple[Dict[str, Any]]:
        """Get optimizer parameters for this layer.

        Args:
            lr (float):

        Returns:
            Tuple[Dict[str, Any]]:
        """
        return [
            {
                "params": list(self.latent_mapper.parameters())
                + list(self.seq2vec.parameters()),
                "lr": lr * self.mapping_network_lr_coef,
            },
            {
                "params": list(self.features.parameters()),
                "lr": lr,
            },
        ]

    def to_uint8(self, x: torch.Tensor) -> torch.Tensor:
        """Map the output of the generator model to uint8.

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
        """
        x = x / 2 + 1 / 2
        return (x * 255.0).type(torch.uint8)


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
            nn.Sequential(
                EqualLinear(latent_dim, w_dim), nn.LeakyReLU(negative_slope=0.1)
            ),
        )
        for _ in range(7):
            self.mapper.append(
                nn.Sequential(
                    EqualLinear(w_dim, w_dim),
                    nn.LeakyReLU(negative_slope=0.1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map x to w plane.

        Args:
            x (torch.Tensor): Input 2D tensor.

        Returns:
            torch.Tensor: Mapped 2D tensor.
        """
        x = F.normalize(x, dim=-1)
        return self.mapper(x)


class GeneratorBlock(nn.Module):
    """Generator block module."""

    GeneratorResult = namedtuple("GeneratorResult", "x,rgb")

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        upsample: bool,
    ) -> None:
        """Intiailizes a generator block.

        Args:
            in_channels (int):
            out_channels (int):
            w_dim (int):
            upsample (bool):

        Returns:
            None:
        """
        super().__init__()

        self.style1 = StyleBlock(
            w_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.style2 = StyleBlock(
            w_dim,
            upsample=upsample,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        if upsample:
            self.skip_upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.skip_upsample = nn.Identity()

        self.to_rgb = nn.Conv2d(out_channels, 3, 1, 1, 0)

        self.register_buffer("residual_scaler", torch.as_tensor([math.sqrt(0.5)]))

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> GeneratorResult:
        """Forward call of generator block.

        Args:
            x (torch.Tensor): input 4D tensor.
            w (torch.Tensor): input 2D latent tensor.
            rgb (torch.Tensor, optional): residual input. Defaults to None.

        Returns:
            GeneratorResult:
        """
        x = self.style1(x, w)
        x = self.style2(x, w)
        x_rgb = self.to_rgb(x)
        if rgb is None:
            rgb = x_rgb
        else:
            rgb = self.skip_upsample(rgb) + x_rgb
        return GeneratorBlock.GeneratorResult(x=x, rgb=rgb)


class AdditiveNoiseBlock(nn.Module):
    """StyleGAN additive noise block."""

    def __init__(self) -> None:
        """Initialize the additive noise module."""
        super().__init__()
        self.coef = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a weighted view of the center cropped noise.

        Args:
            x (torch.Tensor): 4D feature maps. [B, C, H, W]

        Returns:
            torch.Tensor: Cropped noise.
        """
        noise = torch.empty(
            x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device
        ).normal_()
        noise = noise * self.coef
        x = noise + x
        return x


class StyleBlock(nn.Module):
    """Style block module."""

    def __init__(
        self, w_dim: int, eps: float = None, upsample: bool = False, **conv_params
    ) -> None:
        """Intiailize a generator block module.

        Args:
            w_dim (int): Fine grained latent dimension.
            output_shape (Tuple[int, int]): The dimensions of the output of the block.
            eps (float, optional): Defaults to None. (Value controlled by layer default eps.)
            upsample (bool, optional): Whether to upsample the input. Defaults to False.
        """
        super().__init__()

        self.mod_conv = ModulatedConv2D(eps=eps, **conv_params)
        self.B = AdditiveNoiseBlock()
        self.A = EqualLinear(w_dim, conv_params["in_channels"])

        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear") if upsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply a modulated block for the generator.

        Args:
            x (torch.Tensor): Input 4D tensor.
            w (torch.Tensor): Input fine-grained latent 2D tensor.

        Returns:
            torch.Tensor: torch.Tensor
        """
        x = self.upsample(x)

        scores = self.A(w)
        # add conv and add bias
        x = self.mod_conv(x, scores)
        # add noise
        x = self.B(x)

        return self.activation(x)

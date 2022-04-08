"""Generator module."""
from collections import namedtuple
from typing import Any, Dict, Tuple, Union

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
                output_dim=self.conv_channels[0],
            ),
            nn.Unflatten(dim=-1, unflattened_size=(self.conv_channels[0], 1, 1)),
            nn.Upsample(scale_factor=4),
        )

        self.latent_mapper = LatentMapper(self.latent_dim, self.w_dim)

        self.features = nn.ModuleList(
            [GeneratorHead(conv_channels[0], conv_channels[0], self.w_dim)]
        )
        for in_ch, out_ch in zip(self.conv_channels, self.conv_channels[1:]):
            self.features.append(
                GeneratorBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    w_dim=self.w_dim,
                    upsample=True,
                    add_noise=True,
                ),
            )

        self.rescale_rgb = nn.Tanh()

    def forward(
        self,
        z: torch.Tensor,
        title_seq: BatchEncoding,
        stochastic_noise: torch.Tensor,
        return_w: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Map a noise and a sequence to an image.

        Args:
            z (torch.Tensor): Latent noise input.
            title_seq (torch.Tensor): Title sequence.
            stochastic_noise (torch.Tensor): The stochastic input noise.
            return_w (bool, optional): Whether to return w or not.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: If return_w is False, only returns the rgb output, else returns rgb and w in the same order as mentioned as a tuple.
        """
        if stochastic_noise.ndim == 3:
            stochastic_noise = stochastic_noise.unsqueeze(1)
        # enriching the latent vector
        w = self.latent_mapper(z)
        # embed the sequence to a vector
        x = self.seq2vec(title_seq)
        rgb = None
        for f in self.features:
            x, rgb = f(x, w, stochastic_noise, rgb)
        rgb = self.rescale_rgb(rgb)
        if return_w:
            return rgb, w
        else:
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
                EqualLinear(latent_dim, w_dim), nn.LeakyReLU(negative_slope=0.2)
            ),
        )
        for _ in range(7):
            self.mapper.append(
                nn.Sequential(
                    EqualLinear(w_dim, w_dim),
                    nn.LeakyReLU(negative_slope=0.2),
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


GeneratorBlockResult = namedtuple("GeneratorResult", "x,rgb")


class GeneratorHead(nn.Module):
    """Generator head block for translation of text represenation into image."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
    ) -> None:
        """Initialize a GeneratorHead object.

        Args:
            in_channels (int):
            out_channels (int):
            w_dim (int):

        Returns:
            None:
        """
        super().__init__()
        self.style = StyleBlock(
            w_dim=w_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            add_noise=False,
            demod=True,
        )
        self.to_rgb = StyleBlock(
            w_dim=w_dim,
            in_channels=out_channels,
            out_channels=3,
            kernel_size=1,
            padding=0,
            add_noise=False,
            demod=False,
        )

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, *args, **kwargs
    ) -> GeneratorBlockResult:
        """Get the base generated 4D input.

        Args:
            x (torch.Tensor): 4D text representation.
            w (torch.Tensor): 2D latent representation.

        Returns:
            GeneratorBlockResult:
        """
        x = self.style(x, w, None)
        rgb = self.to_rgb(x, w, None)
        return x, rgb


class GeneratorBlock(nn.Module):
    """Generator block module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        w_dim: int,
        upsample: bool,
        add_noise: bool,
    ) -> None:
        """Intiailizes a generator block.

        Args:
            n_layers: number of style layers.
            in_channels (int): input channels for each style layer.
            out_channels (int): output channels for each style layer.
            w_dim (int): The input dimension of w.
            upsample (bool): Whether to upsample or not.
            add_noise (bool): Whether to add noise or not.

        Returns:
            None:
        """
        super().__init__()

        self.style1 = StyleBlock(
            w_dim=w_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            add_noise=add_noise,
            demod=True,
        )
        self.style2 = StyleBlock(
            w_dim=w_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            add_noise=add_noise,
            demod=True,
        )

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            self.upsample = nn.Identity()

        self.to_rgb = StyleBlock(
            w_dim=w_dim,
            in_channels=out_channels,
            out_channels=3,
            kernel_size=1,
            padding=0,
            add_noise=False,
            demod=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        stochastic_noise: torch.Tensor,
        rgb: torch.Tensor = None,
    ) -> GeneratorBlockResult:
        """Forward call of generator block.

        Args:
            x (torch.Tensor): input 4D tensor.
            w (torch.Tensor): input 2D latent tensor.
            rgb (torch.Tensor, optional): residual input. Defaults to None.

        Returns:
            GeneratorBlockResult:
        """
        x = self.upsample(x)
        x = self.style1(x, w, stochastic_noise)
        x = self.style2(x, w, stochastic_noise)
        x_rgb = self.to_rgb(x, w, stochastic_noise)
        rgb = self.upsample(rgb)
        rgb = rgb + x_rgb
        return GeneratorBlockResult(x=x, rgb=rgb)


class AdditiveNoiseBlock(nn.Module):
    """StyleGAN additive noise block."""

    def __init__(self) -> None:
        """Initialize the additive noise module."""
        super().__init__()
        self.scale_noise = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor, stochastic_noise: torch.Tensor) -> torch.Tensor:
        """Compute a weighted view of the center cropped noise.

        Args:
            x (torch.Tensor): 4D feature maps. [B, C, H, W]
            stochastic_noise: 3D noise input with shape [B, 1, FULL_H, FULL_W].

        Returns:
            torch.Tensor: Cropped noise.
        """
        if stochastic_noise.size(1) != 1 and stochastic_noise.ndim != 4:
            raise RuntimeError("Stochastic noise should be 4D and have only 1 channel.")
        noise = stochastic_noise[..., : x.size(2), : x.size(3)]
        noise = noise * self.scale_noise[None, :, None, None]
        x = noise + x
        return x


class DoNotAddNoise(nn.Module):
    """A dummy layer to indicate whether noise should be added or not."""

    def forward(self, x: torch.Tensor, stochastic_noise: torch.Tensor) -> torch.Tensor:
        """Return the image and do nothing with the noise.

        Args:
            x (torch.Tensor):
            stochastic_noise (torch.Tensor):

        Returns:
            torch.Tensor: The input image.
        """
        return x


class StyleBlock(nn.Module):
    """Style block module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        w_dim: int,
        eps: float = None,
        add_noise: bool = False,
        demod: bool = True,
    ) -> None:
        """Intiailize a generator block module.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The kernel size used in convolution.
            padding (int): The padding used in the convolution.
            w_dim (int): Fine grained latent dimension.
            eps (float, optional): Defaults to None. (Value controlled by layer default eps.)
            add_noise (bool, optional): Whether to add stochastic noise to the input in this layer.
            demod (bool, optional): Whether to apply demodulation or not in the modulated convolution.
        """
        super().__init__()

        self.mod_conv = ModulatedConv2D(
            in_channels, out_channels, kernel_size, padding, eps=eps, demod=demod
        )
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.B = AdditiveNoiseBlock() if add_noise else DoNotAddNoise()
        self.A = EqualLinear(w_dim, in_channels, bias=1)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, stochastic_noise: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply a modulated block for the generator.

        Args:
            x (torch.Tensor): Input 4D tensor.
            w (torch.Tensor): Input fine-grained latent 2D tensor.
            stochastic_noise (torch.Tensor): The 4D stochastic input noise with shape [B, 1, H, W].

        Returns:
            torch.Tensor: torch.Tensor
        """
        if stochastic_noise is None and not isinstance(self.B, DoNotAddNoise):
            raise RuntimeError(
                "Cannot input None as stochastic noise when 'add_noise' is set to True"
                " for the style block."
            )
        scores = self.A(w)
        # add conv and add bias
        x = self.mod_conv(x, scores)
        # add noise and bias
        x = self.B(x, stochastic_noise) + self.bias.reshape(1, -1, 1, 1)

        return self.activation(x)

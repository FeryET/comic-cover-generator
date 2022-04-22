"""Sequence to vector module."""

from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForSequenceClassification, BatchEncoding

from comic_cover_generator.ml.constants import Constants
from comic_cover_generator.ml.model.base import EqualLinear
from comic_cover_generator.typing import TypedDict


class Seq2VecParams(TypedDict):
    """Parameters of Seq2Vec model."""

    transformer_model: str


class Seq2Vec(nn.Module):
    """A layer which maps a sequence of varying length to vectors."""

    def __init__(self, transformer_model: str, output_dim: int) -> None:
        """Initialize a seq2vec instance.

        Args:
            transformer_model (str):
            output_channels (int):
            output_dim (int):

        Returns:
            None:
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            transformer_model, cache_dir=Constants.cache_dir
        )
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_model, cache_dir=Constants.cache_dir
        )
        self.transformer.classifier = nn.Sequential(
            EqualLinear(self.config.hidden_size, output_dim, bias=True),
            nn.LeakyReLU(0.2),
        )

        for p in self.transformer.parameters():
            p.requires_grad = False

        for p in self.transformer.classifier.parameters():
            p.requires_grad = True

    def forward(self, seq: BatchEncoding) -> Tensor:
        """Map sequence to vector.

        Args:
            seq (BatchEncoding):

        Returns:
            Tensor:
        """
        x = self.transformer(**seq).logits
        return x

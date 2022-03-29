import random
import string
from collections.abc import Sequence

import pytest
import torch
from matplotlib import collections

from comic_cover_generator.ml.utils import collate_fn


@pytest.fixture(params=[(42, 2, 8, 8), (32, 1, 4, 16), (1, 13, 16, 32)])
def batch(request):
    random_seed, batch_size, img_size, max_seq_len = request.param
    generator = torch.Generator().manual_seed(random_seed)
    images = torch.rand(batch_size, 3, img_size, img_size, generator=generator)
    titles_seq = [
        torch.randint(0, 256, (torch.randint(2, max_seq_len, (1,)),))
        for _ in range(batch_size)
    ]
    full_titles = [
        "".join(random.choices(string.ascii_uppercase + string.digits, k=max_seq_len))
    ]
    return [
        {"image": img, "title_seq": t_seq, "full_title": title}
        for img, t_seq, title in zip(
            images,
            titles_seq,
            full_titles,
        )
    ]


def test_collate_fn_pass(batch):
    result = collate_fn(batch)
    assert result["image"].size(0) == len(result["title_seq"])
    assert isinstance(result["image"], torch.Tensor)
    assert isinstance(result["title_seq"], Sequence) and not isinstance(
        result["title_seq"], torch.Tensor
    )
    assert "full_title" not in result

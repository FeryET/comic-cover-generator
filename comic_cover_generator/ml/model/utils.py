"""Model utility module."""

import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """Initialize weights of a module.

    Args:
        m (nn.Module): module.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

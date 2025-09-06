# -*- coding: utf-8 -*-
"""PlaceHolder: base_optimizer.

This file provides pytorch native optimizer and custom optimizer.
"""


import logging
from typing import Dict
from typing import List

from torch.optim import ASGD
from torch.optim import LBFGS
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import Adamax
from torch.optim import AdamW
from torch.optim import Optimizer


logger = logging.getLogger(__name__)


def build_optimizer(  # pylint: disable=too-many-branches
    type: str,
    params_policy: List[Dict],
    enable_bn_weight_decay: bool = True,
    **kwargs,  # pylint: disable=redefined-builtin
) -> Optimizer:
    """Build optimizer."""
    # use decay_mult
    if not kwargs.get("use_fsdp", False):
        for group in params_policy:
            group["weight_decay"] = group["decay_mult"] * kwargs["weight_decay"]

        if not enable_bn_weight_decay:
            logger.warning(" ! Weight decay NOT applied to BN parameters ")
            for group in params_policy:
                if "BN" in group["name"]:
                    group["weight_decay"] = 0

            for group in params_policy:
                logger.warning(
                    "[build_optimizer]: group: %s has %d params, lr_mult: %f,"
                    " decay_mult: %f. weight_decay: %f",
                    group["name"],
                    len(group["params"]),
                    group["lr_mult"],
                    group["decay_mult"],
                    group["weight_decay"],
                )
    if "use_fsdp" in kwargs:
        kwargs.pop("use_fsdp")

    if type == "ASGD":
        # reference: https://pytorch.org/docs/stable/optim.html
        return ASGD(params_policy, **kwargs)
    if type == "LBFGS":
        # reference: https://pytorch.org/docs/stable/optim.html
        return LBFGS(params_policy, **kwargs)
    if type == "SGD":
        # reference: https://pytorch.org/docs/stable/optim.html
        return SGD(params_policy, **kwargs)
    if type == "Adam":
        # reference: https://pytorch.org/docs/stable/optim.html
        return Adam(params_policy, **kwargs)
    if type == "Adamax":
        # reference: https://pytorch.org/docs/stable/optim.html
        return Adamax(params_policy, **kwargs)
    if type == "AdamW":
        # reference: https://pytorch.org/docs/stable/optim.html
        return AdamW(params_policy, **kwargs)
    raise ValueError(f"optimizer {type} not supported!!!")

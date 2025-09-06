# -*- coding: utf-8 -*-
"""ComposeWeightedLoss."""

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch

from unirobot.brain.model.loss.base_loss import BaseLoss
from unirobot.utils.unirobot_slot import LOSS


class ComposeWeightedLoss(BaseLoss):
    """ComposeWeightedLoss."""

    def __init__(
        self,
        loss_cfg_list=List[Dict[str, Union[float, Dict[str, Any]]]],
        name: str = "ComposeWeightedLoss",
    ) -> None:
        """Initialize."""
        super().__init__(name=name)
        self._loss_func_list: List[Tuple[float, BaseLoss]] = []

        if loss_cfg_list is None:
            raise ValueError(
                "If enable train mode, expect loss_cfg_list is not None, but got None."
            )

        if not isinstance(loss_cfg_list, list):
            raise ValueError(
                f"Expect type(loss_cfg_list)=list but got {type(loss_cfg_list)}."
            )

        for _loss_func_cfg in loss_cfg_list:
            loss_weight = _loss_func_cfg.pop("loss_weight", 1.0)
            loss_func = LOSS.build(_loss_func_cfg.pop("loss_cfg"))
            if not isinstance(loss_func, BaseLoss):
                raise ValueError(
                    f"Expect type(loss_func)=BaseLoss, but got {type(loss_func)}."
                )

            self._loss_func_list.append((loss_weight, loss_func))

    def forward(
        self,
        model_outputs: Dict[str, Any],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Loss."""
        total_loss_dict: Dict[str, Any] = {}
        for (loss_weight, loss_func) in self._loss_func_list:
            loss_dict = loss_func(model_outputs, model_inputs)

            for loss_name, loss_value in loss_dict.items():
                if isinstance(loss_value, torch.Tensor):
                    loss_dict[loss_name] *= loss_weight
                else:
                    raise TypeError(f"{loss_name} is not a tensor.")

            if len(set(total_loss_dict.keys()) & set(loss_dict.keys())) != 0:
                raise ValueError(
                    f"Dict keys conflict.\n"
                    f"total_loss_dict keys: {total_loss_dict.keys()}.\n"
                    f"loss_dict keys: {loss_dict.keys()}."
                )

            total_loss_dict.update(**loss_dict)

        return total_loss_dict

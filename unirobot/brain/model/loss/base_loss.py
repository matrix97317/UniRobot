# -*- coding: utf-8 -*-
"""UniRobot BaseLoss."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch


class BaseLoss(torch.nn.Module, ABC):
    """BaseLoss Abstract Interface.

    Args:
        name (bool): Module name.
        **kwargs (Dict[str, Any]): Extra inputs for torch.nn.Module.
    """

    def __init__(
        self,
        name: str = "base_loss",
        **kwargs,
    ) -> None:
        """Init base loss."""
        super().__init__(**kwargs)
        self._name = name

    @abstractmethod
    def forward(
        self,
        model_outputs: Dict[str, Any],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Forward loss.

        Args:
            model_outputs (Dict[str, Any]): Pred result of model.
            model_inputs (Dict[str, Any]): Ground truth.

        Returns:
            Loss dict.
        """
        raise NotImplementedError()

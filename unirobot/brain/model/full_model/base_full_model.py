# -*- coding: utf-8 -*-
"""UniRobot BaseFullModel."""

from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

import torch


def _forward_unimplemented(*args, **kwargs) -> Any:
    """Unimplemented forward."""
    raise NotImplementedError(
        f"Unimplemented forward triggered.\n"
        f"The number of positional arguments: {len(args)}.\n"
        f"The number of keyword arguments: {len(kwargs)}."
    )


class BaseFullModel(torch.nn.Module, ABC):
    """Full Model Interface.

    Args:
        sub_module_cfg (Dict[str, Any]): Config dict of sub module.
        train_mode (bool): Whether to enable training. Default=`True`.
    """

    infer_forward: Callable[..., Tuple[Any, ...]] = _forward_unimplemented
    train_forward: Callable[..., Dict[str, Any]] = _forward_unimplemented

    def __init__(
        self,
        sub_module_cfg: Dict[str, Any],
        train_mode: bool = True,
        **kwargs,
    ) -> None:
        """Init base task model."""
        super().__init__(**kwargs)
        if sub_module_cfg is None:
            raise ValueError("sub_module_cfg value is None.")
        self._train_mode = train_mode
        self._batch_data = None

    def set_mode(
        self,
        train_mode: bool = True,
    ) -> None:
        """Set train mode.

        Args:
            train_mode (bool): Whether to enable training mode.
        """
        self._train_mode = train_mode

    def init_weight(self) -> None:
        """Init model components weight."""
        raise NotImplementedError()

    def set_batch_data(self, batch_data):
        """Set batch data."""
        self._batch_data = batch_data

    def get_batch_data(self):
        """Return batch data."""
        return self._batch_data

    def parse_inputs_data(self, inputs_data):  # pylint: disable=no-self-use
        """Parse inputs data."""
        return inputs_data

    def wrap_outputs_data(self, outputs_data):  # pylint: disable=no-self-use
        """Wrap outputs data."""
        return outputs_data

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        """Forward base module.

        Args:
            *args, **kwargs: Inputs of forward.

        Returns:
            If `train_mode`=True, return training outputs.
            If `train_mode`=False, return infer outputs.
        """
        if self._train_mode:
            return self.train_forward(*args, **kwargs)

        return self.infer_forward(*args, **kwargs)

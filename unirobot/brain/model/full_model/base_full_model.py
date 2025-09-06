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

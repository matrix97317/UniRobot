# -*- coding: utf-8 -*-
"""PlaceHolder: base_lr_scheduler.

This file provides both native and custom lr schedulers.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Optional

import numpy as np
from torch.optim import Optimizer


class BaseLrScheduler(ABC):
    """LR Scheduler base class."""

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        warmup_type: Optional[str] = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
    ) -> None:
        """LR scheduler changes learning rate during training.

        Args:
            optimizer (torch.optim.Optimizer): params optimizer. Default: None.
            warmup_type (string): Type of warmup used. It can be None(use no warmup),
                'constant', 'linear' or 'exp'.  Default: None.
            warmup_iters (int): The number of iterations or epochs that warmup
                lasts. Default: 0.
            warmup_ratio (float): LR used at the beginning of warmup equals to
                warmup_ratio * initial_lr. Default: 0.1.
        """
        if optimizer is None:
            raise ValueError("you don't give optimizer in BaseLrScheduler.")
        # validate the "warmup" argument
        if warmup_type is not None:
            if warmup_type not in ["constant", "linear", "exp"]:
                raise ValueError(
                    f'"{warmup_type}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear", "exp"'
                )
            if warmup_iters <= 0:
                raise ValueError("`warmup_iters` must be a positive integer")
            if not 0 < warmup_ratio <= 1.0:
                raise ValueError('"warmup_ratio" must be in range (0,1]')

        self._optimizer = optimizer
        self._warmup_type = warmup_type
        self._warmup_iters = warmup_iters
        self._warmup_ratio = warmup_ratio

        self._base_lr: Any = None  # save all init learning rate form optimizer.
        self._regular_lr: list = (
            []
        )  # expected learing rate if no warming up is performed

    def _set_lr(self, lr_groups: Union[dict, list]):
        """Set learning rate into optimizer's param groups."""
        if isinstance(self._optimizer, dict):
            if not isinstance(lr_groups, dict):
                raise ValueError(
                    f"Expect type(lr_groups)=dict\
                     when type(optimizer)=dict, but got {type(lr_groups)}.",
                )
            if len(self._optimizer.param_groups) != len(lr_groups):
                raise ValueError(
                    f"optimizer params group \
                    {len(self._optimizer.param_groups)} \
                        != base lr group {len(lr_groups)}"
                )
            for k, optim in self._optimizer.items():
                for param_group, learning_rate in zip(optim.param_groups, lr_groups[k]):
                    param_group["lr"] = learning_rate
        else:
            if not isinstance(lr_groups, list):
                raise ValueError(
                    f"Expect type(lr_groups)=list\
                     when type(optimizer)!=dict, but got {type(lr_groups)}.",
                )
            optimizer_param_group_size = len(self._optimizer.param_groups)
            if optimizer_param_group_size != len(lr_groups):
                raise ValueError(
                    f"optimizer params group \
                        {optimizer_param_group_size} \
                            != base lr group {len(lr_groups)}"
                )
            for param_group, learning_rate in zip(
                self._optimizer.param_groups,
                lr_groups,
            ):
                param_group["lr"] = learning_rate

    @abstractmethod
    def get_lr(self, cur_epoch: int, base_lr: Any) -> Any:
        """Get learning rate based on\
             `base_lr` `and cur_epoch`."""
        raise NotImplementedError()

    def get_regular_lr(self, cur_epoch: int) -> Any:
        """Get learning rate based optimizer's attribute."""
        if isinstance(self._optimizer, dict):
            lr_groups = {}
            for k in self._optimizer.keys():
                _lr_group = [
                    self.get_lr(cur_epoch, _base_lr) for _base_lr in self._base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups

        return [self.get_lr(cur_epoch, _base_lr) for _base_lr in self._base_lr]

    def get_warmup_lr(self, cur_iters: int) -> Any:
        """Return warmup learning rate of cur_iters \
          in different ways."""

        def _get_warmup_lr(cur_iters, regular_lr):
            if self._warmup_type == "constant":
                warmup_lr = [_lr * self._warmup_ratio for _lr in regular_lr]
            elif self._warmup_type == "linear":
                k = (1 - cur_iters / self._warmup_iters) * (1 - self._warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self._warmup_type == "exp":
                k = self._warmup_ratio ** (1 - cur_iters / self._warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self._regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self._regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups

        return _get_warmup_lr(cur_iters, self._regular_lr)

    def init_base_lr(self) -> None:
        """Init base learning rate from optimizer."""
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(self._optimizer, dict):
            self._base_lr = {}
            for k, optim in self._optimizer.items():
                for group in optim.param_groups:
                    if "lr_mult" in group:
                        group.setdefault("initial_lr", group["lr"] * group["lr_mult"])
                    else:
                        group.setdefault("initial_lr", group["lr"])
                _base_lr = [group["initial_lr"] for group in optim.param_groups]
                self._base_lr.update({k: _base_lr})
        else:
            for group in self._optimizer.param_groups:
                if "lr_mult" in group:
                    group.setdefault("initial_lr", group["lr"] * group["lr_mult"])
                else:
                    group.setdefault("initial_lr", group["lr"])
            self._base_lr = [
                group["initial_lr"] for group in self._optimizer.param_groups
            ]

    def step(self, cur_iter: int, cur_epoch: int) -> None:
        """Set learning rate.

        Args:
            cur_iter (int): set learning rate in warmup process by step way.
            cur_epoch (int): set learning rate \
                    in regular training process by epoch way.
        """
        self._regular_lr = self.get_regular_lr(cur_epoch)
        if self._warmup_type is None or cur_iter >= self._warmup_iters:
            self._set_lr(self._regular_lr)
        else:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(warmup_lr)

    @abstractmethod
    def state_dict(self) -> Dict:
        """Return LR Scheduler State."""
        raise NotImplementedError()

    @abstractmethod
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict is used init params."""
        raise NotImplementedError()


class EpochLrScheduler(BaseLrScheduler):
    """LR Scheduler in epoch by epoch way."""

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        decay_boundary: Optional[Union[int, List[int]]] = None,
        gamma: float = 0.1,
        min_lr: float = 0.000001,
        warmup_type: Optional[str] = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
        **kwargs,
    ) -> None:
        """Step LR scheduler with min_lr clipping.

        Args:
            optimizer (torch.optim.Optimizer): params optimizer.
                Default: None.
            decay_boundary (int | list[int]): epoch to decay the LR.
                If an int value is given,regard it as the decay interval.
                If a list is given, decay LR at
                these epoch value. Default: None.
            gamma (float, optional): Decay LR ratio. Default: 0.1.
            min_lr (float, optional): Minimum LR value to keep. If LR after decay
                is lower than `min_lr`, it will be clipped to this value. If None
                is given, we don't perform lr clipping. Default: None.
            warmup_type (string): Type of warmup used. It can be None(use no warmup),
                'constant', 'linear' or 'exp'. Default: None.
            warmup_iters (int): The number of iterations that warmup
                lasts. Default: 0.
            warmup_ratio (float): LR used at the beginning of warmup equals to
                warmup_ratio * initial_lr. Default: 0.
        """
        if isinstance(decay_boundary, list):
            for epoch_value in decay_boundary:
                if epoch_value <= 0:
                    raise ValueError("epoch_value must > 0.")
        elif isinstance(decay_boundary, int):
            if decay_boundary <= 0:
                raise ValueError("epoch_value must > 0.")
        else:
            raise TypeError(
                "`decay_boundary` must be a positive integer or a list of positive "
                "integers."
            )
        self._decay_boundary = decay_boundary
        self._gamma = gamma
        self._min_lr = min_lr
        super().__init__(
            optimizer,
            warmup_type,
            warmup_iters,
            warmup_ratio,
            **kwargs,
        )

    def get_lr(self, cur_epoch, base_lr) -> float:
        """Get cur_epoch learning rate based cur_epoch."""
        progress = cur_epoch
        # calculate exponential term
        if isinstance(self._decay_boundary, int):
            exp = progress // self._decay_boundary
        else:
            exp = len(self._decay_boundary)
            for i, step_value in enumerate(self._decay_boundary):
                if progress < step_value:
                    exp = i
                    break
        learning_rate = base_lr * (self._gamma**exp)
        if self._min_lr is not None:
            # clip to a minimum value
            learning_rate = max(learning_rate, self._min_lr)
        return learning_rate

    def state_dict(self) -> Dict:
        """Return LR Scheduler State."""
        return {
            "warmup_type": self._warmup_type,
            "warmup_iters": self._warmup_iters,
            "warmup_ratio": self._warmup_ratio,
            "base_lr": self._base_lr,
            "regular_lr": self._regular_lr,
            "decay_boundary": self._decay_boundary,
            "gamma": self._gamma,
            "min_lr": self._min_lr,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict is used init params."""
        self._warmup_type = state_dict["warmup_type"]
        self._warmup_iters = state_dict["warmup_iters"]
        self._warmup_ratio = state_dict["warmup_ratio"]
        self._base_lr = state_dict["base_lr"]
        self._regular_lr = state_dict["regular_lr"]
        self._decay_boundary = state_dict["decay_boundary"]
        self._gamma = state_dict["gamma"]
        self._min_lr = state_dict["min_lr"]


class CosineLrScheduler(BaseLrScheduler):
    """LR Scheduler in cosine way."""

    def __init__(
        self,
        optimizer: Optional[Optimizer] = None,
        total_epoch: int = 1,
        min_lr: float = 0.000001,
        warmup_type: Optional[str] = None,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.1,
        **kwargs,
    ) -> None:
        """Step LR scheduler with min_lr clipping.

        Args:
            optimizer (torch.optim.Optimizer): params optimizer.
                Default: None.
            total_epoch (int, optional): the total epochs of training process.
                Default: 1.
            min_lr (float, optional): Minimum LR value to keep. If LR after decay
                is lower than `min_lr`, it will be clipped to this value. If None
                is given, we don't perform lr clipping. Default: None.
            warmup_type (string): Type of warmup used. It can be None(use no warmup),
                'constant', 'linear' or 'exp'. Default: None.
            warmup_iters (int): The number of iterations that warmup
                lasts. Default: 0.
            warmup_ratio (float): LR used at the beginning of warmup equals to
                warmup_ratio * initial_lr. Default: 0.
        """
        if isinstance(total_epoch, int):
            if total_epoch < 1:
                raise ValueError("total_epoch must >= 1.")
        else:
            raise TypeError("`total_epoch` must be a positive integer and >=1.")
        self._total_epoch = total_epoch
        self._min_lr = min_lr
        super().__init__(
            optimizer,
            warmup_type,
            warmup_iters,
            warmup_ratio,
            **kwargs,
        )

    def get_lr(self, cur_epoch, base_lr) -> float:
        """Get cur_epoch learning rate based cur_epoch."""
        learning_rate = self._min_lr + (
            0.5
            * (1 + np.cos(np.pi * cur_epoch / self._total_epoch))
            * (base_lr - self._min_lr)
        )

        return learning_rate

    def state_dict(self) -> Dict:
        """Return LR Scheduler State."""
        return {
            "warmup_type": self._warmup_type,
            "warmup_iters": self._warmup_iters,
            "warmup_ratio": self._warmup_ratio,
            "base_lr": self._base_lr,
            "regular_lr": self._regular_lr,
            "total_epoch": self._total_epoch,
            "min_lr": self._min_lr,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict is used init params."""
        self._warmup_type = state_dict["warmup_type"]
        self._warmup_iters = state_dict["warmup_iters"]
        self._warmup_ratio = state_dict["warmup_ratio"]
        self._base_lr = state_dict["base_lr"]
        self._regular_lr = state_dict["regular_lr"]
        self._total_epoch = state_dict["total_epoch"]
        self._min_lr = state_dict["min_lr"]

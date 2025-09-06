# -*- coding: utf-8 -*-
"""base trainer."""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable

from unirobot.utils.cfg_parser import PyConfig


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """BaseTrainer provides based training workflow.

    Args:
        cfg (PyConfig): cfg is consist of experiment paramters.
            `unirobot.utils.cfg_parser.PyConfig` parse exp_xxx.py as cfg(Dict).
    """

    def __init__(
        self,
        cfg: PyConfig,
    ) -> None:
        """Init BaseTrainer based config dict."""
        self._cfg = cfg

    @abstractmethod
    def enable_deterministic(self) -> None:
        """Enable Algorithm."""
        raise NotImplementedError()

    @abstractmethod
    def load_ckpt(self) -> None:
        """Load ckpt or resume ckpt."""
        raise NotImplementedError()

    @abstractmethod
    def train(self) -> None:
        """Training Flow."""
        raise NotImplementedError()

    @abstractmethod
    def train_one_epoch(
        self,
        epoch: int,
    ) -> None:
        """One epoch training."""
        raise NotImplementedError()

    @abstractmethod
    def train_one_step(
        self,
        epoch: int,
        step: int,
    ) -> Any:
        """One step training."""
        raise NotImplementedError()

    @abstractmethod
    def register_hook(
        self,
        func_name: str,
        func: Callable,
    ) -> None:
        """Register custom function, that is inserted into training flow."""
        raise NotImplementedError()

    @abstractmethod
    def get_model(self) -> Any:
        """Return model is used to training."""
        raise NotImplementedError()

    @abstractmethod
    def get_optimizer(self) -> Any:
        """Return optimizer is used to training."""
        raise NotImplementedError()

    @abstractmethod
    def get_lr_scheduler(self) -> Any:
        """Return lr scheduler is used to training."""
        raise NotImplementedError()

    @abstractmethod
    def get_dataloader(self) -> Any:
        """Return dataloader is used to training."""
        raise NotImplementedError()

# -*- coding: utf-8 -*-
"""Robot interface."""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


logger = logging.getLogger(__name__)

ROBOT_MODE = ["teleoperation", "model_local", "model_server"]


class BaseRobot(ABC):
    """The abstract interface of Robot.

    Args:
        mode (str): Mode in ['teleoperation','model_local','model_server'].
        sensor_cfg : Config sensors.
        motor_cfg : Config motors.
        teleoperator_cfg : Config remote controller.
        model_cfg : Config model.
    """

    def __init__(
        self,
        mode: str = "teleopreation",
        sensor_cfg: Optional[Dict] = None,
        motor_cfg: Optional[Dict] = None,
        teleoperator_cfg: Optional[Dict] = None,
        model_cfg: Optional[Dict] = None,
    ):
        """Init."""
        self._mode = mode
        if self._mode not in ROBOT_MODE:
            raise ValueError(
                f"current robot mode {self._mode} not be supported, refer to {ROBOT_MODE}"
            )
        self._sensor_cfg = sensor_cfg
        self._motor_cfg = sensor_cfg
        self._teleoperator_cfg = teleoperator_cfg
        self._model_cfg = model_cfg

    def get_observation(self, *args, **kwargs) -> None:
        """Get env info from sensors."""
        raise NotImplementedError()

    def set_action(self, *args, **kwargs) -> None:
        """Send action seq to motors."""
        raise NotImplementedError()

    def run_teleoperation(self, *args, **kwargs) -> None:
        """Run teleoperation mode."""
        raise NotImplementedError()

    def run_model(self, *args, **kwargs) -> None:
        """Run model inference mode."""
        raise NotImplementedError()

    def run_rl(self, *args, **kwargs) -> None:
        """Run reinforcement learning."""
        raise NotImplementedError()

    def run(self, *args, **kwargs) -> None:
        """Run robot."""
        raise NotImplementedError()

    def pre_process(self, *args, **kwargs) -> None:
        """Run model pre process."""
        raise NotImplementedError()

    def post_process(self, *args, **kwargs) -> None:
        """Run model post process."""
        raise NotImplementedError()

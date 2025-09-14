# -*- coding: utf-8 -*-
"""Robot interface."""
import logging
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional


logger = logging.getLogger(__name__)

ROBOT_MODE = ["teleoperation", "model_local", "model_server"]


def _forward_unimplemented(*args, **kwargs) -> Any:
    """Unimplemented forward."""
    raise NotImplementedError(
        f"Unimplemented forward triggered.\n"
        f"The number of positional arguments: {len(args)}.\n"
        f"The number of keyword arguments: {len(kwargs)}."
    )


class BaseRobot(ABC):
    """The abstract interface of Robot.

    Args:
        mode (str): Mode in ['teleoperation','model_local','model_server'].
        sensor_cfg : Config sensors.
        motor_cfg : Config motors.
        teleoperator_cfg : Config remote controller.
        model_cfg : Config model.
    """

    forward: Callable[..., Tuple[Any, ...]] = _forward_unimplemented

    def __init__(
        self,
        mode: str = "teleopreation",
        use_rl: bool = False,
        sensor_cfg: Optional[Union[Dict[str, Dict[str, Any]], Dict[str, Any]]] = None,
        motor_cfg: Optional[Union[Dict[str, Dict[str, Any]], Dict[str, Any]]] = None,
        teleoperator_cfg: Optional[
            Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
        ] = None,
        model_cfg: Optional[Dict] = None,
        **kwargs,
    ):
        """Init."""
        super().__init__(**kwargs)
        self._mode = mode
        self._use_rl = use_rl
        if self._mode not in ROBOT_MODE:
            raise ValueError(
                f"current robot mode {self._mode} not be supported, refer to {ROBOT_MODE}"
            )
        self._sensor_cfg = sensor_cfg
        self._motor_cfg = motor_cfg
        self._teleoperator_cfg = teleoperator_cfg
        self._model_cfg = model_cfg

    def __str__(self):
        """Return str."""
        return f"{self.__class__.__name__}"

    def get_observation(self, *args, **kwargs) -> None:
        """Get env info from sensors."""
        return self.forward(*args, **kwargs)

    def get_teleoperator(self, *args, **kwargs) -> None:
        """Get env info from teleoperator."""
        return self.forward(*args, **kwargs)

    def set_action(self, *args, **kwargs) -> None:
        """Send action seq to motors."""
        return self.forward(*args, **kwargs)

    def run_teleoperation(self, *args, **kwargs) -> None:
        """Run teleoperation mode."""
        return self.forward(*args, **kwargs)

    def run_model(self, *args, **kwargs) -> None:
        """Run model inference mode."""
        return self.forward(*args, **kwargs)

    def run_rl(self, *args, **kwargs) -> None:
        """Run reinforcement learning."""
        return self.forward(*args, **kwargs)

    def run(self, *args, **kwargs) -> None:
        """Run robot."""
        return self.forward(*args, **kwargs)

    def pre_process(self, *args, **kwargs) -> None:
        """Run model pre process."""
        return self.forward(*args, **kwargs)

    def post_process(self, *args, **kwargs) -> None:
        """Run model post process."""
        return self.forward(*args, **kwargs)

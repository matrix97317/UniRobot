# -*- coding: utf-8 -*-
"""SoArm101-Follower Motor."""
import logging
from typing import Any
from typing import Dict
import os
import time

from pydantic import BaseModel

from unirobot.robot.device_interface import BaseDevice
from unirobot.robot.ros_lib.motors_bus import Motor, MotorCalibration, MotorNormMode
from unirobot.robot.ros_lib.feetech import FeetechMotorsBus, OperatingMode
from unirobot.robot.motor.so_arm101_calib import so_arm_101_follower_calib


logger = logging.getLogger(__name__)


class CalibConfig(BaseModel):
    """Calibration config."""

    motors: Dict[str, MotorCalibration]


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]],
    max_relative_target: dict[str, float],
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""
    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError(
                "max_relative_target keys must match those of goal_present_pos."
            )
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe. %s\n",
            warnings_dict,
        )

    return safe_goal_positions


class SoArm101Follower(BaseDevice):
    """The abstract interface of Robot Device.

    Args:
        host_name (str): Device's ID, such as IP.
        port (str) : Device' Port, such as IP's port, UART prot.
    """

    def __init__(
        self,
        host_name: str = "so_arm101_follower",
        port: str = "1234",
        use_degrees: bool = False,
        max_relative_target: int = None,
        disable_torque_on_disconnect: bool = True,
    ):
        """Init."""
        super().__init__(host_name=host_name, port=port)
        # self._host_name = host_name
        # self._port = port
        # os.chmod(self._port, 0o777)  # 设置权限为 rwxr-xr-x
        self.use_degrees = use_degrees
        self.max_relative_target = max_relative_target
        self.disable_torque_on_disconnect = disable_torque_on_disconnect
        norm_mode_body = (
            MotorNormMode.DEGREES if self.use_degrees else MotorNormMode.RANGE_M100_100
        )
        self.calibration = self.load_calibration()
        self.bus = FeetechMotorsBus(
            port=self._port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration.motors,
        )

    def load_calibration(self):
        """Load calibration."""
        return CalibConfig(motors=so_arm_101_follower_calib)

    def open(self, *args, **kwargs) -> None:
        """Open a device."""
        if self.bus.is_connected:
            raise RuntimeError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    def configure(self, *args, **kwargs) -> None:
        """Configure a device."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 0)

    def get(self, *args, **kwargs) -> None:
        """Get info from device."""
        if not self.bus.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        return obs_dict

    def put(self, action: dict[str, Any]) -> dict[str, Any]:
        """Put info to device."""
        if not self.bus.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.max_relative_target
            )

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def close(self, *args, **kwargs) -> None:
        """Close a device."""
        if not self.bus.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        self.bus.disconnect(self.disable_torque_on_disconnect)
        logger.info(f"{self} disconnected.")

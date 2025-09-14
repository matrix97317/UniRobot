# -*- coding: utf-8 -*-
"""Robot's Device interface."""
import logging
from typing import Dict
import os
import time

from pydantic import BaseModel

from unirobot.robot.device_interface import BaseDevice
from unirobot.robot.ros_lib.motors_bus import Motor, MotorCalibration, MotorNormMode
from unirobot.robot.ros_lib.feetech import FeetechMotorsBus, OperatingMode
from unirobot.robot.teleoperator.so_arm101_calib import so_arm_101_leader_calib


logger = logging.getLogger(__name__)


class CalibConfig(BaseModel):
    """Calibration config."""

    motors: Dict[str, MotorCalibration]


class SoArm101Leader(BaseDevice):
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
    ):
        """Init."""
        # self._host_name = host_name
        # self._port = port
        super().__init__(host_name=host_name, port=port)
        # os.chmod(self._port, 0o777)  # 设置权限为 rwxr-xr-x
        self.use_degrees = use_degrees
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
        return CalibConfig(motors=so_arm_101_leader_calib)

    def open(self, *args, **kwargs) -> None:
        """Open a device."""
        if self.bus.is_connected:
            raise RuntimeError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    def configure(self, *args, **kwargs) -> None:
        """Configure a device."""
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get(self, *args, **kwargs) -> None:
        """Get info from device."""
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def put(self, *args, **kwargs) -> None:
        """Put info to device."""
        raise NotImplementedError()

    def close(self, *args, **kwargs) -> None:
        """Close a device."""
        if not self.bus.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

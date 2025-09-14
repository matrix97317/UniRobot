# -*- coding: utf-8 -*-
"""Robot So101."""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import os
import select
import sys
import time
import shutil
import h5py
import numpy as np

from unirobot.utils.unirobot_slot import SENSOR
from unirobot.utils.unirobot_slot import MOTOR
from unirobot.utils.unirobot_slot import TELEOPERATOR
from unirobot.robot.robot_interface import BaseRobot


logger = logging.getLogger(__name__)

ROBOT_MODE = ["teleoperation", "model_local", "model_server"]


def enter_pressed(key):
    """Check if a specific key is pressed."""
    return (
        select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline().strip() == key
    )


class So101(BaseRobot):
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
        use_rl: bool = False,
        sensor_cfg: Optional[Union[Dict[str, Dict[str, Any]], Dict[str, Any]]] = None,
        motor_cfg: Optional[Union[Dict[str, Dict[str, Any]], Dict[str, Any]]] = None,
        teleoperator_cfg: Optional[
            Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
        ] = None,
        model_cfg: Optional[Dict] = None,
        fps: int = 25,
        dataset_dir: str = "./so101_dataset",
        task_name: str = "default",
    ):
        """Init."""
        super().__init__(
            mode=mode,
            use_rl=use_rl,
            sensor_cfg=sensor_cfg,
            motor_cfg=motor_cfg,
            teleoperator_cfg=teleoperator_cfg,
            model_cfg=model_cfg,
        )
        self._top_sensor = SENSOR.build(self._sensor_cfg["top"])
        self._hand_sensor = SENSOR.build(self._sensor_cfg["hand"])
        self._motor = MOTOR.build(self._motor_cfg)
        self._teleoper = TELEOPERATOR.build(self._teleoperator_cfg)
        self._fps = fps
        self._count_episode = 0
        self._dataset_dir = dataset_dir
        self._task_name = task_name
        self._colloct_data: dict = {"top_img": [], "hand_img": [], "action": []}
        self._start_colloct = False
        self._frame_count = 0

    def get_observation(self, *args, **kwargs) -> Any:
        """Get env info from sensors."""
        top_img = self._top_sensor.get_async()
        hand_img = self._hand_sensor.get_async()
        return {"top_img": top_img, "hand_img": hand_img}

    def get_teleoperator(self, *args, **kwargs) -> Any:
        """Get env info from teleoperator."""
        return {"motor_info": self._teleoper.get()}

    def set_action(self, *args, **kwargs) -> Any:
        """Send action seq to motors."""
        return self._motor.put(kwargs["action"])

    def run_teleoperation(self, *args, **kwargs) -> Any:
        """Run teleoperation mode."""
        try:
            self._top_sensor.open()
            self._hand_sensor.open()
            self._motor.open()
            self._teleoper.open()
            while True:
                loop_start = time.perf_counter()
                sensor_info = self.get_observation()
                tele_info = self.get_teleoperator()
                actual_action = self.set_action(action=tele_info["motor_info"])
                dt_s = time.perf_counter() - loop_start
                time.sleep(1 / self._fps - dt_s)
                loop_s = time.perf_counter() - loop_start
                logger.info("Action: %s", actual_action)
                logger.info(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                if self._start_colloct:
                    logger.info("Collect Frame ID: %s", self._frame_count)
                    self._colloct_data["top_img"].append(sensor_info["top_img"])
                    self._colloct_data["hand_img"].append(sensor_info["hand_img"])
                    self._colloct_data["action"].append(
                        np.array(list(actual_action.values()))
                    )
                    self._frame_count += 1
                if enter_pressed("s"):
                    if not self._start_colloct:
                        self._frame_count = 0
                        self._start_colloct = True
                    else:
                        self._start_colloct = False
                        self._count_episode += 1
                        self._save_data()
                        self._colloct_data = {
                            "top_img": [],
                            "hand_img": [],
                            "action": [],
                        }
                        logger.info("Save Episode ID: %s", self._count_episode)
                        time.sleep(1)
            # self._motor.close()
            # self._top_sensor.close()
            # self._hand_sensor.close()
            # self._teleoper.close()
        except Exception as e:
            logger.error("So101 occur error: %s", e)
            self._motor.close()
            self._top_sensor.close()
            self._hand_sensor.close()
            self._teleoper.close()
        except KeyboardInterrupt:
            logger.error("Close Robot %s", self)
            self._motor.close()
            self._top_sensor.close()
            self._hand_sensor.close()
            self._teleoper.close()

        logger.info("Close Robot %s", self)

    def run_model(self, *args, **kwargs) -> None:
        """Run model inference mode."""
        if self._mode == "teleoperation":
            self.run_teleoperation()

    def run_rl(self, *args, **kwargs) -> None:
        """Run reinforcement learning."""
        raise NotImplementedError()

    def run(self, *args, **kwargs) -> None:
        """Run robot."""
        if self._mode == "teleoperation":
            self.run_teleoperation()

    def pre_process(self, *args, **kwargs) -> None:
        """Run model pre process."""
        raise NotImplementedError()

    def post_process(self, *args, **kwargs) -> None:
        """Run model post process."""
        raise NotImplementedError()

    def _save_data(
        self,
    ):
        task_data_dir = os.path.join(self._dataset_dir, self._task_name)
        if os.path.exists(task_data_dir) and self._count_episode == 0:
            shutil.rmtree(task_data_dir)
        os.makedirs(task_data_dir, exist_ok=True)
        episode_file = os.path.join(
            task_data_dir, f"episode_{self._count_episode:04d}.hdf5"
        )
        top_imgs = np.stack(self._colloct_data["top_img"], axis=0)
        hand_imgs = np.stack(self._colloct_data["hand_img"], axis=0)
        action = np.stack(self._colloct_data["action"], axis=0)

        with h5py.File(episode_file, "w") as f:
            # 创建数据集
            f.create_dataset(
                "top_imgs", data=top_imgs, compression="gzip", compression_opts=4
            )
            f.create_dataset(
                "hand_imgs", data=hand_imgs, compression="gzip", compression_opts=4
            )
            f.create_dataset(
                "action", data=action, compression="gzip", compression_opts=4
            )
            # 添加元数据
            f.attrs["frames_num"] = top_imgs.shape[0]
        logger.warning("Save data to %s", episode_file)

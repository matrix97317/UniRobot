# -*- coding: utf-8 -*-
"""Robot So101."""
import os
import select
import sys
import time
import shutil
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from threading import Thread, Event, Lock
from queue import Queue
import json

import cv2
import h5py
import numpy as np

from unirobot.utils.unirobot_slot import SENSOR
from unirobot.utils.unirobot_slot import MOTOR
from unirobot.utils.unirobot_slot import TELEOPERATOR
from unirobot.robot.robot_interface import BaseRobot
from unirobot.brain.data.dataset.act_dataset import ACTDataset
from unirobot.robot.utils.http_client import HTTPPolicyClient


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
        episode_format: str = "episode_{:d}.hdf5",
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
        self._camera_names = ["top", "hand"]
        self._motor = MOTOR.build(self._motor_cfg)
        self._teleoper = TELEOPERATOR.build(self._teleoperator_cfg)
        self._fps = fps
        self._count_episode = 0
        self._dataset_dir = dataset_dir
        self._task_name = task_name
        self._episode_format = episode_format
        self._colloct_data: dict = {
            "top": [],
            "hand": [],
            "obs_action": [],
            "action": [],
        }
        self._start_colloct = False
        self._frame_count = 0
        self.init_dataset()
        # save data thread
        self._data_queue = Queue(maxsize=10)
        self._data_thread = Thread(
            target=self._save_data,
            args=(
                self._dataset_dir,
                self._task_name,
                self._camera_names,
                self._count_episode,
                self._episode_format,
            ),
            daemon=True,
        )
        if self._mode == "model_server":
            self._client = HTTPPolicyClient(self._model_cfg["base_url"])
            self._start_infer = False

    def init_dataset(self, *args, **kwargs) -> None:
        """Init dataset."""
        task_data_dir = os.path.join(self._dataset_dir, self._task_name)
        if not os.path.exists(task_data_dir):
            os.makedirs(task_data_dir, exist_ok=True)
        else:
            meta_file = os.path.join(task_data_dir, "meta.json")
            if not os.path.exists(meta_file):
                logger.error("Dataset dir exist but not meta.json %s", meta_file)
                raise FileExistsError(
                    f"Dataset dir exist but not meta.json {meta_file}"
                )

            with open(meta_file, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            self._count_episode = meta_data["count_episode"]
            logger.info(
                "Init dataset dir: %s, count episode: %s ",
                self._dataset_dir,
                self._count_episode,
            )

    def get_observation(self, *args, **kwargs) -> Any:
        """Get env info from sensors."""
        top_img = self._top_sensor.get_async()
        hand_img = self._hand_sensor.get_async()
        return {"top": top_img, "hand": hand_img}

    def get_teleoperator(self, *args, **kwargs) -> Any:
        """Get env info from teleoperator."""
        return {"motor_info": self._teleoper.get()}

    def get_robot_state(self, *args, **kwargs) -> Any:
        """Get env info from teleoperator."""
        return {"motor_info": self._motor.get()}

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
            self._data_thread.start()
            while True:
                loop_start = time.perf_counter()
                sensor_info = self.get_observation()
                robot_state = self.get_robot_state()
                tele_info = self.get_teleoperator()
                actual_action = self.set_action(action=tele_info["motor_info"])
                dt_s = time.perf_counter() - loop_start
                if (1 / self._fps - dt_s) < 0:
                    logger.warning(
                        f"Run So101 too slow: {dt_s*1e3:.2f}ms, expect {(1/self._fps)*1e3:.2f}ms"
                    )
                    time.sleep(1 / self._fps)
                else:
                    time.sleep(1 / self._fps - dt_s)
                loop_s = time.perf_counter() - loop_start
                logger.info("Action: %s", actual_action)
                logger.info(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                if self._start_colloct:
                    logger.info("Collect Frame ID: %s", self._frame_count)
                    self._colloct_data["top"].append(sensor_info["top"])
                    self._colloct_data["hand"].append(sensor_info["hand"])
                    # self._colloct_data["obs_action"].append(
                    #     np.array(list(tele_info["motor_info"].values()))
                    # )
                    self._colloct_data["obs_action"].append(
                        np.array(list(robot_state["motor_info"].values()))
                    )
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
                        self._data_queue.put(self._colloct_data)
                        self._colloct_data = {
                            "top": [],
                            "hand": [],
                            "obs_action": [],
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
            logger.error("Starting Close Robot..... %s", self)
            logger.warning("Process the remaining data")
            self._data_queue.join()
            self._motor.close()
            self._top_sensor.close()
            self._hand_sensor.close()
            self._teleoper.close()
            logger.warning("All data processed")
            logger.info("start compute norm stats")
            self._compte_norm_stats()
            logger.error("Finished Close Robot %s", self)

        logger.info("Close Robot %s", self)

    def run_model_server(
        self,
    ) -> None:
        """Run model server."""
        try:
            self._top_sensor.open()
            self._hand_sensor.open()
            self._motor.open()
            self._teleoper.open()
            init_pos_cnt = 0
            gripper_frame_cnt =[]
            griper = 0
            while True:
                loop_start = time.perf_counter()
                sensor_info = self.get_observation()
                robot_state = None
                human_action = None
                model_action = None
                if self._start_infer:
                    robot_state = self.get_robot_state()
                    data = {}
                    qpos = np.array(list(robot_state["motor_info"].values()))

                    if qpos[5]>2.5 and qpos[5]<3.0:
                        qpos[5]=qpos[5]-1.63
                    if qpos[5]>9 and qpos[5]<10:
                        qpos[5]=qpos[5]-8.33
                   
                    
                    data["qpos"] = qpos
                    _, top_img = cv2.imencode(
                        ".jpg", sensor_info["top"], [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    )
                    _, hand_img = cv2.imencode(
                        ".jpg", sensor_info["hand"], [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    )
                    data["hand"] = top_img
                    data["top"] = hand_img
                    s = time.perf_counter()
                    output = self._client.msgpack_infer(data)
                    e = time.perf_counter()
                    logger.info(f"cost time {(e-s)*1000} ms")
                    logger.info("Model Infer Action: %s", output["action"])
                    griper=output["action"][5]

                    # if  output["action"][5]>30:
                    #     griper=output["action"][5]-20
                    # else:
                    #     griper=output["action"][5]
                    #     gripper_frame_cnt.append(output["action"][5])
                    # if len(gripper_frame_cnt)>80:
                    #     griper=1.4
                    #     gripper_frame_cnt=[]
                    # else:
                    #     

                    model_action = output["action"]
                    if init_pos_cnt >100:
                        set_motor_info = {
                            "shoulder_pan.pos": output["action"][0],
                            "shoulder_lift.pos": output["action"][1],
                            "elbow_flex.pos": output["action"][2],
                            "wrist_flex.pos": output["action"][3],
                            "wrist_roll.pos": output["action"][4],
                            "gripper.pos":griper,
                        }
                    else:
                        set_motor_info={'shoulder_pan.pos': -6.387665198237897,
                        'shoulder_lift.pos': -53.157674613132585, 
                        'elbow_flex.pos': 31.609977324263014, 
                        'wrist_flex.pos': 89.33901918976545, 
                        'wrist_roll.pos': 9.290187891440496, 
                        'gripper.pos': 0.7961783439490446
                        }
                    _ = self.set_action(action=set_motor_info)
                    init_pos_cnt+=1

                if not self._start_infer:
                    robot_state = self.get_teleoperator()
                    human_action = self.set_action(action=robot_state["motor_info"])
                    logger.info("Human Infer Action: %s", human_action)

                dt_s = time.perf_counter() - loop_start
                if (1 / self._fps - dt_s) < 0:
                    logger.warning(
                        f"Run So101 too slow: {dt_s*1e3:.2f}ms, expect {(1/self._fps)*1e3:.2f}ms"
                    )
                    time.sleep(1 / self._fps)
                else:
                    time.sleep(1 / self._fps - dt_s)
                loop_s = time.perf_counter() - loop_start
                logger.info(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
                if self._use_rl:
                    logger.info("Collect Frame ID: %s", self._frame_count)
                    self._colloct_data["top"].append(sensor_info["top"])
                    self._colloct_data["hand"].append(sensor_info["hand"])
                    self._colloct_data["obs_action"].append(
                        np.array(list(robot_state["motor_info"].values()))
                    )
                    self._colloct_data["model_action"].append(model_action)
                    self._colloct_data["human_action"].append(human_action)

                    self._frame_count += 1
                if enter_pressed("s"):
                    if not self._start_infer:
                        self._frame_count = 0
                        self._start_infer = True
                        init_pos_cnt = 0
                    else:
                        self._start_infer = False
                        self._count_episode += 1
                        # self._data_queue.put(self._colloct_data)
                        # self._colloct_data = {
                        #     "top": [],
                        #     "hand": [],
                        #     "obs_action": [],
                        #     "action": [],
                        # }
                        # logger.info("Save Episode ID: %s", self._count_episode)
                        # time.sleep(1)
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
            logger.error("Starting Close Robot..... %s", self)
            # logger.warning("Process the remaining data")
            # self._data_queue.join()
            self._motor.close()
            self._top_sensor.close()
            self._hand_sensor.close()
            self._teleoper.close()
            # logger.warning("All data processed")
            # logger.info("start compute norm stats")
            # self._compte_norm_stats()
            logger.error("Finished Close Robot %s", self)

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
        if self._mode == "model_server":
            self.run_model_server()

    def pre_process(self, *args, **kwargs) -> None:
        """Run model pre process."""
        raise NotImplementedError()

    def post_process(self, *args, **kwargs) -> None:
        """Run model post process."""
        raise NotImplementedError()

    def _compte_norm_stats(self) -> None:
        """Compute norm stats for dataset."""
        task_data_dir = os.path.join(self._dataset_dir, self._task_name)

        if not os.path.exists(task_data_dir):
            logger.error("Dataset dir not exist: %s", task_data_dir)
            raise FileNotFoundError(f"Dataset dir not exist: {task_data_dir}")

        if not os.path.exists(os.path.join(task_data_dir, "meta.json")):
            logger.error(
                "Dataset dir not exist meta.json: %s",
                os.path.join(task_data_dir, "meta.json"),
            )
            raise FileNotFoundError(
                f"Dataset dir not exist meta.json: {os.path.join(task_data_dir, 'meta.json')}"
            )

        with open(os.path.join(task_data_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        ACTDataset.get_norm_stats(task_data_dir, meta_data["count_episode"])

    def _save_data(
        self,
        dataset_dir: str,
        task_name: str,
        camera_names: List[str],
        count_episode: int,
        episode_format: str,
    ):
        """Save data to dataset."""
        try:
            curr_episode = count_episode
            task_data_dir = os.path.join(dataset_dir, task_name)
            while True:
                data = self._data_queue.get()
                # compress and pad images
                if data is None:
                    continue
                all_encoded = []
                img_data_dict = {}
                for cam in camera_names:
                    key = f"/observations/images/{cam}"
                    encoded_list = []
                    for img in data[cam]:
                        _, enc = cv2.imencode(
                            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                        )
                        encoded_list.append(enc)
                        all_encoded.append(len(enc))
                    img_data_dict[key] = encoded_list

                padded_size = max(all_encoded)

                for cam in camera_names:
                    key = f"/observations/images/{cam}"
                    padded = [
                        np.pad(enc, (0, padded_size - len(enc)), constant_values=0)
                        for enc in img_data_dict[key]
                    ]
                    img_data_dict[key] = padded

                # save episode
                episode_file = os.path.join(
                    task_data_dir, episode_format.format(curr_episode)
                )
                frame_cnt = len(data[camera_names[0]])

                with h5py.File(episode_file, "w", rdcc_nbytes=1024**2 * 2) as root:
                    obs_dict = root.create_group("observations")
                    image = obs_dict.create_group("images")

                    for cam_name in camera_names:
                        img_shape = (frame_cnt, padded_size)
                        img_chunk = (1, padded_size)

                        image.create_dataset(
                            cam_name, img_shape, "uint8", chunks=img_chunk
                        )

                        image[cam_name][...] = img_data_dict[
                            f"/observations/images/{cam_name}"
                        ]

                    state_dim = 6
                    obs_dict.create_dataset("qpos", (frame_cnt, state_dim))
                    root.create_dataset("action", (frame_cnt, state_dim))
                    obs_dict["qpos"][...] = data["obs_action"]
                    root["action"][...] = data["action"]
                curr_episode += 1
                # save meta
                with open(
                    os.path.join(task_data_dir, "meta.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump({"count_episode": curr_episode}, f)
                logger.warning("Save data to %s", episode_file)

                self._data_queue.task_done()
        except Exception as e:
            logger.error("So101 save data occur error: %s", e)

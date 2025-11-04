# -*- coding: utf-8 -*-
"""Dataset of ACT Model."""
import logging
import os
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import h5py
import torch
import numpy as np
from PIL import Image

from unirobot.brain.data.dataset.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


class ACTDataset(BaseDataset):
    """Dataset for ACT data.

    Args:
        mode (str): Mode in _VALID_MODE.
        meta_file (Dict[str, Dict[str, List[str]]] or None) : Meta file.
        task_name (str or None) : Meta file key, expect str.
        robot_name (str or None) : Meta file key, expect str.

    """

    DATASET_META_FILE = {
        "robot_name": {
            "task_name": {
                "num_episodes": 10,
                "episode_format": "episode_{:d}.hdf5",
                "train": ["/path/task_name-dir/"],
                "val": ["/path/task_name-dir/"],
                "norm_stats": ["/path/task_name-dir/norm_stats.pkl"],
            }
        }
    }

    def __init__(
        self,
        mode: str,
        meta_file: Optional[Dict[str, Dict[str, List[str]]]] = None,
        task_name: str = "pick_toy",
        robot_name: str = "so_arm101",
        camera_names: Optional[List[str]] = None,
        seed: int = 666,
        train_ratio: float = 0.8,
        sample_full_episode: bool = False,
        transforms: Union[List[Any], Dict[str, Any], None] = None,
    ) -> None:
        """Init."""
        super().__init__(
            mode=mode,
            transforms=transforms,
        )
        meta_file = meta_file or self.DATASET_META_FILE
        self._meta_file = meta_file[robot_name][task_name]  # type: ignore[assignment]
        self._dataset_dir_paths = self._meta_file[self._mode]  # type: ignore[index]
        self._num_episodes = self._meta_file["num_episodes"]
        self._episode_format = self._meta_file["episode_format"]
        self._norm_stats_file = self._meta_file["norm_stats"]
        self._norm_stats = self.load_norm_stats()

        self._train_ratio = train_ratio
        self._sample_full_episode = sample_full_episode
        self._camera_names = camera_names
        np.random.seed(seed)
        rng = np.random.default_rng(seed=seed)
        self._cache_records = []
        if self._mode == "train":
            self._cache_records = rng.permutation(self._num_episodes)[
                : int(self._train_ratio * self._num_episodes)
            ]
        if self._mode == "val":
            self._cache_records = rng.permutation(self._num_episodes)[
                int(self._train_ratio * self._num_episodes) :
            ]

        logger.info(
            "Data path: %s   num_episodes: [%d].",
            self._dataset_dir_paths,
            self._num_episodes,
        )
        logger.info("%s index: %s.", self._mode, self._cache_records)
        logger.info("norm stats: %s", self._norm_stats)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, int]]:
        """Get one item.

        Args:
            idx (int): The index.

        Returns:
            One data item, which contains `image` and `gt`.
        """
        # get 1 episode data
        current_episode_id = self._cache_records[idx]
        hdf5_file = os.path.join(
            self._dataset_dir_paths, self._episode_format.format(current_episode_id)
        )
        qpos = None
        # qvel = None
        aligned_action_shape = (550, 6)
        aligned_episode_len = 550
        aligned_action = None
        with h5py.File(hdf5_file, "r") as root:
            # original_action_shape = (370,6) #root["/action"].shape
            # episode_len = 370#original_action_shape[0]
            episode_len = root["/action"].shape[0]
            if self._sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(root["/action"].shape[0])
            if start_ts+50>=episode_len:
                start_ts=start_ts
                if start_ts==episode_len-1:
                    start_ts=episode_len-2
            else:
                start_ts=start_ts+50
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            # qvel = root["/observations/qvel"][start_ts]
            # print(f"------ oringinal actionshape {root["/action"][()].shape}")
            # print(f"------ oringinal qpos {root["/observations/qpos"][()].shape}")
            image_dict = dict()
            for cam_name in self._camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            # TODO: collect data error: qpos==action
            action = root["/action"][
                min(episode_len-1, start_ts + 1) : episode_len
            ]  # hack, to make timesteps more aligned
            if action.shape[0]>aligned_episode_len:
                aligned_action = action[:aligned_episode_len,:]
            else:
                aligned_action = action

            # TODO: if qpos==action-1,action==action

            # action = root["/action"][
            #     max(0, start_ts - 1) : episode_len
            # ]  # hack, to make timesteps more aligned
            # action_len = episode_len - max(
            #     0, start_ts - 1
            # )  # hack, to make timesteps more aligned

        # print(f"===== action len {action_len}")
        # print(f"===== episode_len {episode_len}")

        padded_action = np.zeros(aligned_action_shape, dtype=np.float32)
        # print(f'+++ padded_action.shape {padded_action.shape}')
        # print(f'+++ action.shape {action.shape}')
        padded_action[: aligned_action.shape[0], :] = aligned_action
        # print(f'++ a padded_action.shape {padded_action.shape}')
        is_pad = np.zeros(aligned_episode_len)
        is_pad[aligned_action.shape[0] :] = 1


        # img aug
        rand_h = np.random.randint(0, 480-101)
        rand_w = np.random.randint(0, 640-101)
        mask = np.zeros((100,100,3))
        rand_brightness = np.random.choice([-1, 1], size=1)
        if rand_brightness==1:
            rand_brightness_value = np.random.randint(30,80)
        else:
            rand_brightness_value = np.random.randint(0,20)
        rand_aug = True if np.random.random()<0.3 else False
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self._camera_names:
            # for frame_cnt in range(image_dict[cam_name].shape[0]):
            #     print(image_dict[cam_name].shape)
            #     breakpoint()
            img = cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)[:, :, ::-1]
            if rand_aug:
                if cam_name=='top':
                    img[rand_h:rand_h+100,rand_w:rand_w+100,:]=mask
                img = cv2.convertScaleAbs(img, alpha=float(rand_brightness), beta=float(rand_brightness_value)) # beta为正，提高亮度
                
            all_cam_images.append(
                img[None,]
            )
            # all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.concatenate(all_cam_images, axis=0)
        # print(all_cam_images.shape)
        # breakpoint()

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (
            action_data - self._norm_stats["action_mean"]
        ) / self._norm_stats["action_std"]
        qpos_data = (qpos_data - self._norm_stats["qpos_mean"]) / self._norm_stats[
            "qpos_std"
        ]
        # print(qpos_data.shape)
        # print(action_data.shape)
        # print(is_pad.shape)
        # breakpoint()

        data_dict = {
            "image": image_data,
            "qpos": qpos_data.squeeze(),
            "actions": action_data,
            "is_pad": is_pad,
        }

        if self._transforms is not None:
            data_dict = self._transforms(data_dict)

        return data_dict

    def preprocess_server_data(self, data: Any) -> Any:
        """Preprocess server data."""
        qpos = data["qpos"]
        all_cam_images = []
        for cam_name in self._camera_names:
            all_cam_images.append(
                cv2.imdecode(data[cam_name], cv2.IMREAD_COLOR)[:, :, ::-1][None,]
            )
        all_cam_images = np.concatenate(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        # action_data = (
        #     action_data - self._norm_stats["action_mean"]
        # ) / self._norm_stats["action_std"]
        qpos_data = (qpos_data - self._norm_stats["qpos_mean"]) / self._norm_stats[
            "qpos_std"
        ]

        data_dict = {
            "image": image_data.unsqueeze(0),
            "qpos": qpos_data,
        }
        if self._transforms is not None:
            data_dict = self._transforms(data_dict)
        return data_dict

    def get_infer_data(self, epsoide_idx) -> Any:
        """Get one item.

        Args:
            idx (int): The index.

        Returns:
            One data item, which contains `image` and `gt`.
        """
        # get 1 episode data
        current_episode_id = self._cache_records[epsoide_idx]
        hdf5_file = os.path.join(
            self._dataset_dir_paths, self._episode_format.format(current_episode_id)
        )
        qpos = None
        # qvel = None
        # original_action_shape = (370,6)
        # episode_len = 370
        with h5py.File(hdf5_file, "r") as root:
            for frame_cnt in range(root["/action"].shape[0]):
                qpos = root["/observations/qpos"][frame_cnt]
                image_dict = dict()
                for cam_name in self._camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                        frame_cnt
                    ]
                action = root["/action"][frame_cnt]

                # new axis for different cameras
                all_cam_images = []
                for cam_name in self._camera_names:
                    # for frame_cnt in range(image_dict[cam_name].shape[0]):
                    #     print(image_dict[cam_name].shape)
                    #     breakpoint()
                    all_cam_images.append(
                        cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)[
                            :, :, ::-1
                        ][
                            None,
                        ]
                    )
                    # all_cam_images.append(image_dict[cam_name])
                all_cam_images = np.concatenate(all_cam_images, axis=0)
                # print(all_cam_images.shape)
                # breakpoint()

                # construct observations
                image_data = torch.from_numpy(all_cam_images)
                qpos_data = torch.from_numpy(qpos).float()
                action_data = torch.from_numpy(action).float()

                # channel last
                image_data = torch.einsum("k h w c -> k c h w", image_data)

                # normalize image and change dtype to float
                image_data = image_data / 255.0
                # action_data = (
                #     action_data - self._norm_stats["action_mean"]
                # ) / self._norm_stats["action_std"]
                qpos_data = (
                    qpos_data - self._norm_stats["qpos_mean"]
                ) / self._norm_stats["qpos_std"]

                data_dict = {
                    "image": image_data.unsqueeze(0),
                    "qpos": qpos_data,
                    "actions": action_data,
                    "frame_cnt": frame_cnt,
                }

                if self._transforms is not None:
                    data_dict = self._transforms(data_dict)
                yield data_dict

    def load_norm_stats(self) -> Any:
        """Load norm stats file."""
        if os.path.exists(self._norm_stats_file):
            with open(self._norm_stats_file, "rb") as fin:
                return pickle.load(fin)
        else:
            logger.warning(
                "%s not exists. will auto generate it.", self._norm_stats_file
            )
            return self.get_norm_stats(self._dataset_dir_paths, self._num_episodes)

    # def parse_meta_file(self) -> List[Tuple[str, int]]:
    #     """Parse data path.

    #     Returns:
    #         cache_records (list): List of {hash_code, gt} for each data.
    #     """
    #     cache_records: List[Tuple[str, int]] = []
    #     for data_path in self._data_paths:
    #         with open(data_path, encoding="utf8") as fin:
    #             for line in fin:
    #                 (
    #                     path
    #                 ) = line.split()
    #                 del path
    #                 cache_records.append(path)
    #     return cache_records

    def __len__(self) -> int:
        """Get length."""
        return len(self._cache_records)

    # @staticmethod
    # def data_reader(data_path: str) -> Image.Image:
    #     """Read image.

    #     Args:
    #         data_path (str): Hash code of image.

    #     Returns:
    #         RGB image object(PIL Image.Image).
    #     """
    #     image_bytes = read_bytes([data_path])[0]
    #     with Image.open(io.BytesIO(image_bytes)) as image:
    #         image = image.convert("RGB")
    #     return image

    @staticmethod
    def get_norm_stats(dataset_dir, num_episodes):
        """Get norm stats from dataset."""
        all_qpos_data = []
        all_action_data = []
        init_pos_data = []
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                # qvel = root["/observations/qvel"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            init_pos_data.append(torch.from_numpy(action)[:50, :])
        all_qpos_data = torch.cat(all_qpos_data)
        all_action_data = torch.cat(all_action_data)
        all_init_pos_data = torch.cat(init_pos_data)
        # all_action_data = all_action_data
        init_pos_mean = all_init_pos_data.mean(
            dim=[
                0,
            ],
            keepdim=True,
        )

        # normalize action data
        action_mean = all_action_data.mean(
            dim=[
                0,
            ],
            keepdim=True,
        )
        action_std = all_action_data.std(
            dim=[
                0,
            ],
            keepdim=True,
        )
        action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

        # normalize qpos data
        qpos_mean = all_qpos_data.mean(
            dim=[
                0,
            ],
            keepdim=True,
        )
        qpos_std = all_qpos_data.std(
            dim=[
                0,
            ],
            keepdim=True,
        )
        qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

        stats = {
            "action_mean": action_mean.numpy(),
            "action_std": action_std.numpy(),
            "qpos_mean": qpos_mean.numpy(),
            "qpos_std": qpos_std.numpy(),
            "init_pos_mean": init_pos_mean.numpy(),
            "example_qpos": qpos,
        }
        with open(os.path.join(dataset_dir, "norm_stats.pkl"), "wb") as fout:
            pickle.dump(stats, fout)
        logger.info("[%s] 's norm stats save at current dataset dir.", dataset_dir)
        return stats

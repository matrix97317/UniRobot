# -*- coding: utf-8 -*-
"""Robot HTTP Client."""
import os
import logging
import time

import h5py
import requests

from unirobot.utils.msgpack_numpy import Packer, unpackb
from unirobot.brain.utils.vis_open_loop import plot_dimension_comparison

logger = logging.getLogger(__name__)


class HTTPPolicyClient:
    """Http Client."""

    def __init__(
        self, base_url: str = "https://u691691-9c18-54f8ef79.bjb1.seetacloud.com:8443"
    ):
        self.base_url = base_url
        self.session = requests.Session()
        self.packer = Packer()

        if self.health_check():
            logger.warning("Test Health OK!")
        else:
            logger.error("Test Health Failed!")

    def health_check(self) -> bool:
        """Check Health."""
        try:
            response = self.session.get(f"{self.base_url}/healthz", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def msgpack_infer(self, obs: dict) -> dict:
        """MsgPack Inference."""
        # 将数据打包为 msgpack
        packed_data = self.packer.pack(obs)

        response = self.session.post(
            f"{self.base_url}/msgpack_infer",
            data=packed_data,
            headers={"Content-Type": "application/msgpack"},
            timeout=30,
        )
        response.raise_for_status()

        # 解析 msgpack 响应
        return unpackb(response.content)


if __name__ == "__main__":

    data_list = []
    client = HTTPPolicyClient(
        base_url="https://u2691691-9c18-54f8ef79.bjb1.seetacloud.com:8443"
    )

    with h5py.File(
        "/home/matrix97317/workspace/UniRobot/so101_dataset/pick_toy2/episode_0.hdf5",
        "r",
    ) as root:
        for frame_cnt in range(root["/action"].shape[0]):
            data = {}
            qpos = root["/observations/qpos"][frame_cnt]
            data["qpos"] = qpos
            data["hand"] = root["/observations/images/hand"][frame_cnt]
            data["top"] = root["/observations/images/top"][frame_cnt]
            action = root["/action"][frame_cnt]
            s = time.perf_counter()
            output = client.msgpack_infer(data)
            e = time.perf_counter()
            print(f"cost time {(e-s)*1000} ms")
            print(output)

            data_list.append({"gt": action, "model": output["action"]})
        plot_dimension_comparison(
            data_list,
            figsize=(20, 25),
            dims=6,
            save_path=os.path.join(
                "./",
                "open_loop_comparison.jpg",
            ),
            dpi=300,
        )

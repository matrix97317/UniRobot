# -*- coding: utf-8 -*-
"""base infer."""
import os
import time
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional

import torch
import numpy as np

from unirobot.utils.cfg_parser import PyConfig
from unirobot.brain.infra.checkpoint_util import CheckpointUtil
from unirobot.utils.file_util import FileUtil
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_rank,
)
from unirobot.utils.unirobot_slot import FULL_MODEL
from unirobot.utils.unirobot_slot import DATASET
from unirobot.brain.utils.vis_open_loop import plot_dimension_comparison
from unirobot.brain.utils.filter_algo import SimpleKalmanFilter
from unirobot.brain.utils.http_server import FastAPIHTTPPolicyServer

logger = logging.getLogger(__name__)


class BaseInfer:
    """BaseInfer provides based infer workflow.

    Args:
        cfg (PyConfig): cfg is consist of experiment paramters.
            `unirobot.utils.cfg_parser.PyConfig` parse exp_xxx.py as cfg(Dict).
        run_name (str, optional): run name. Defaults to None.
        resume (bool, optional): resume ckpt. Defaults to False.
        infer_type (str, optional): infer type. Defaults to None.
        export_type (str, optional): export type. Defaults to None.
        eval_ckpt_list (list, optional): eval ckpt list. Defaults to None.
        test_open_loop (bool, optional): test open loop. Defaults to False.

    """

    def __init__(
        self,
        cfg: PyConfig,
        run_name: Optional[str] = None,
        resume: bool = False,
        infer_type: Optional[str] = None,
        export_type: Optional[str] = None,
        eval_ckpt_list: Optional[list] = None,
        use_kf: bool = False,
        infer_chunk_step: int = 0,
        host_addr: str = "127.0.0.1",
        host_port: int = 6008,
        val_idx:int =0,
    ) -> None:
        """Init BaseInfer based config dict."""
        self._cfg = cfg
        self._model = None
        self._eval_ckpt_list = eval_ckpt_list
        self._ckpt_path = None
        self._ckpt_manager = None
        self._use_interval = False
        self._dataset = None
        self._infer_type = infer_type
        self._export_type = export_type
        self._use_kf = use_kf
        self._infer_chunk_step = infer_chunk_step
        self._infer_cnt = 0
        self._val_idx =val_idx

        self.build_dataset()
        self.build_model()
        self.load_ckpt()
        logger.warning("infer_type: %s", self._infer_type)
        logger.warning("infer_chunk_step: %d", self._infer_chunk_step)
        server_metadata = {
            "server_name": "FastAPI HTTP Policy Server - Binary Support",
            "version": "1.0.0",
            "capabilities": ["msgpack_inference"],
            "supported_formats": ["msgpack", "binary"],
            "max_batch_size": 10,
            "binary_support": True,
        }
        self._model_server = FastAPIHTTPPolicyServer(
            policy=self.model_infer,
            host=host_addr,
            port=host_port,
            metadata=server_metadata,
            infer_chunk_step=infer_chunk_step,
            use_kf=use_kf
        )

    def build_dataset(
        self,
    ) -> None:
        """Build dataset."""
        self._cfg.dataloader["dataset_cfg"]["mode"] = "val"
        self._dataset = DATASET.build(self._cfg.dataloader.dataset_cfg)

    def build_model(
        self,
    ) -> None:
        """Build model."""
        self._cfg.model_flow.full_model_cfg.train_mode = False
        self._model = FULL_MODEL.build(self._cfg.model_flow.full_model_cfg)
        self._model.cuda()

    def load_ckpt(
        self,
    ) -> None:
        """Load ckpt or resume ckpt.

        Args:
            ckpt_path (str): Path to checkpoint.
        """
        if self._eval_ckpt_list is None:
            raise ValueError(
                "eval_ckpt_list is None, please set eval_ckpt_list in infer cfg."
            )

        ckpt_prefix = f"checkpoint_pipeline_rank_{get_pipeline_model_parallel_rank()}"
        for ckpt in self._eval_ckpt_list:
            if ckpt_prefix in ckpt:
                self._ckpt_path = ckpt
                break
        if self._ckpt_path is None:
            raise ValueError(f" {self._eval_ckpt_list} have not {ckpt_prefix} .")

        self._ckpt_manager = CheckpointUtil(
            model=self._model,
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            save_dir=None,
            pretrain_model=self._ckpt_path,
            resume=False,
            to_cuda=self._cfg.ckpt.to_cuda,
            ckpt2model_json=self._cfg.ckpt.ckpt2model_json,
            trace_mode=False,
            use_interval=self._use_interval,
        )
        self._ckpt_manager.load_model()

    def open_loop_infer(
        self,
    ) -> None:
        """Open loop infer."""
        kl = None
        data_list = []
        self._model.eval()
        infer_cnt = 0
        chunk_action = []
        for idx, data in enumerate(self._dataset.get_infer_data(epsoide_idx=self._val_idx )):
            if data is not None:
                data["actions"] = data["actions"].cuda()
                data["image"] = data["image"].cuda()
                data["qpos"] = data["qpos"].cuda()

                s = time.perf_counter()
                if infer_cnt == 0:
                    output = self._model.infer_forward(data)

                model_action = (
                    output["a_hat"].cpu().detach().numpy()[0, infer_cnt, :]
                    * self._dataset._norm_stats["action_std"]
                ) + self._dataset._norm_stats["action_mean"]

                infer_cnt += 1
                if infer_cnt >= self._infer_chunk_step:
                    infer_cnt = 0

                if self._use_kf:
                    if idx == 0:
                        kl = SimpleKalmanFilter(
                            process_variance=0.01,
                            measurement_variance=0.1,
                            initial_position=model_action[0],
                        )
                    kl.predict()
                    filtered_position = kl.update(model_action[0])
                    model_action[0] = filtered_position
                e = time.perf_counter()
                logger.info(f"infer open loop step {idx} time: {(e - s)*1000:.2f} ms")

                data_list.append(
                    {
                        "gt": data["actions"].cpu().detach().numpy(),
                        "model": model_action[0],
                    }
                )
                if idx == 50:
                    for i in range(output["a_hat"][0].shape[0]):
                        chunk_model_action = (
                            output["a_hat"].cpu().detach().numpy()[0, i, :]
                            * self._dataset._norm_stats["action_std"]
                        ) + self._dataset._norm_stats["action_mean"]
                        chunk_action.append(
                            {
                                "gt": chunk_model_action[0],
                                "model": chunk_model_action[0],
                            }
                        )
                del data

        plot_dimension_comparison(
            data_list,
            figsize=(20, 25),
            dims=6,
            save_path=os.path.join(
                FileUtil.get_export_dir(),
                "open_loop_comparison.jpg",
            ),
            dpi=300,
        )
        plot_dimension_comparison(
            chunk_action,
            figsize=(20, 25),
            dims=6,
            save_path=os.path.join(
                FileUtil.get_export_dir(),
                "chunk_action.jpg",
            ),
            dpi=300,
        )

    def model_infer(self, data: Any) -> Any:
        """Model infer function."""
        self._model.eval()
        model_action = None
        with torch.no_grad():
            data = self._dataset.preprocess_server_data(data)
            data["image"] = data["image"].cuda()
            data["qpos"] = data["qpos"].cuda()
            output = self._model.infer_forward(data)
            model_action = (
                output["a_hat"].cpu().detach().numpy()[0, :, :]
                * self._dataset._norm_stats["action_std"]
            ) + self._dataset._norm_stats["action_mean"]

            # self._infer_cnt +=1
            # if self._infer_cnt >= self._infer_chunk_step:
            #     self._infer_cnt = 0

            # if self._use_kf:
            #     if frame_idx == 0:
            #         kl = SimpleKalmanFilter(
            #             process_variance=0.01,
            #             measurement_variance=0.1,
            #             initial_position=model_action[0],
            #         )
            #     kl.predict()
            #     filtered_position = kl.update(model_action[0])
            #     model_action[0] = filtered_position
            return model_action

    def infer(
        self,
    ) -> None:
        """Infer function."""
        if self._infer_type == "open_loop":
            self.open_loop_infer()

        if self._infer_type == "model_server":
            try:
                self._model_server.serve_forever()
            except KeyboardInterrupt:
                print("\n服务器正在关闭...")

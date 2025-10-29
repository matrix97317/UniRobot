# -*- coding: utf-8 -*-
"""base infer."""
import os
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
        test_open_loop: bool = False,
        open_model_server: bool = False,
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
        self._test_open_loop = test_open_loop
        self._open_model_server = open_model_server

        self.build_dataset()
        self.build_model()
        self.load_ckpt()

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

    def infer(
        self,
    ) -> None:
        """Infer function."""
        if self._test_open_loop:
            all_time_actions = torch.zeros([500, 500+40, 6]).cuda()
            last_raw_action = np.zeros((1,6))
            kl = None
            data_list = []
            # self._model.cuda()
            self._model.eval()
            chunk_action = []
            for idx, data in enumerate(self._dataset.get_infer_data(epsoide_idx=1)):
                if data is not None:
                    data["actions"] = data["actions"].cuda()
                    data["image"] = data["image"].cuda()
                    data["qpos"] = data["qpos"].cuda()
                    print(data["actions"].shape)
                    print(data["image"].shape)
                    print(data["qpos"].shape)
                    print(data["frame_cnt"])
                    output = self._model.infer_forward(data)
                    print(output["a_hat"].shape)
                    model_action = (
                        output["a_hat"].cpu().detach().numpy()[0, 0, :]
                        * self._dataset._norm_stats["action_std"]
                    ) + self._dataset._norm_stats["action_mean"]
                    print(model_action.shape)
                    print(output["a_hat"][0].device)
                    # all_time_actions[[idx], idx:idx+40] = output["a_hat"][0]
                    # actions_for_curr_step = all_time_actions[:, idx]
                    # actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    # actions_for_curr_step = actions_for_curr_step[actions_populated]
                    # print("actions_for_curr_step:", actions_for_curr_step.shape)
                    # k = 0.01
                    # exp_weights = np.exp(-k * np.arange(40)[::-1])
                    # exp_weights = exp_weights / exp_weights.sum()
                    # print("exp_weights sum:", exp_weights)
                    # exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    # print("exp_weights:", exp_weights.shape)
                    # raw_action = (output["a_hat"][0] * exp_weights).sum(dim=0, keepdim=True)
                    # print("raw action:", raw_action.shape)
                    # model_action = (
                    #     raw_action.cpu().detach().numpy()
                    #     * self._dataset._norm_stats["action_std"]
                    # ) + self._dataset._norm_stats["action_mean"]
                    # new_raw_action = last_raw_action*0.5+raw_action.cpu().detach().numpy()*0.5
                    # last_raw_action = raw_action.cpu().detach().numpy()

                    # model_action = (
                    #     new_raw_action
                    #     * self._dataset._norm_stats["action_std"]
                    # ) + self._dataset._norm_stats["action_mean"]
                    if idx ==0:
                        kl = SimpleKalmanFilter(
                            process_variance=0.01,
                            measurement_variance=0.1,
                            initial_position=model_action[0],
                        )
                    kl.predict()
                    filtered_position = kl.update(model_action[0])
                    model_action[0] = filtered_position
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

# -*- coding: utf-8 -*-
"""UniRobot checkpoint manager."""

import json
import logging
import os
import shutil
from collections import deque
from functools import wraps
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_rank,
)


logger = logging.getLogger(__name__)


TRAIN_CHECKPOINT_FORMAT = "checkpoint_pipeline_rank_{}_{}.pth.tar"


def rank_zero_only(func):
    """Migrated from pytorch-lightning."""

    @wraps(func)
    def wrapped_fn(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
        if (not dist.is_initialized()) or get_data_parallel_rank() == 0:
            return func(*args, **kwargs)

    return wrapped_fn


def load_checkpoint(
    ckpt_path: str,
    to_cuda: Optional[bool] = False,
) -> Dict[str, Any]:
    """Load checkpoint.

    Args:
        ckpt_path (str): Checkpoint path.
        to_cuda (bool, optional): Whether to remap storage locations to cuda. Read more
            detail at `torch.load.map_location`
            Default=`False`.

    Returns:
        troch.load res.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"File {ckpt_path} does not exist.")

    def _map_location(storage, _):
        if to_cuda:
            device = torch.cuda.current_device()
            return storage.cuda(device)
        return storage

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        logger.warning("dist it not been initialized!")
        rank = 0
    logger.info(
        "==>Rank %d , Pipeline Rank %d, load ckpt %s.",
        rank,
        get_pipeline_model_parallel_rank(),
        ckpt_path,
    )

    return torch.load(ckpt_path, map_location=_map_location, weights_only=False)


class CheckpointUtil:  # pylint: disable=too-many-instance-attributes
    """CheckpointUtil."""

    def __init__(
        self,
        model: Union[nn.Module, nn.parallel.DistributedDataParallel, nn.DataParallel],
        optimizer: Union[Optimizer, None],
        lr_scheduler: Any,  # placeholder: Maybe BaseLrScheduler.
        scaler: Any,  # placeholder: Maybe torch.cuda.amp.GradScaler.
        save_dir: Optional[str] = None,
        use_interval: bool = False,
        pretrain_model: Optional[Union[str, List[str]]] = None,
        enable_load_optimizer: bool = True,
        enable_load_lr_scheduler: bool = True,
        resume: bool = False,
        to_cuda: Optional[bool] = False,
        ckpt2model_json: Optional[str] = None,
        drop_model_params: bool = True,
        trace_mode: bool = False,
    ):
        """CheckpointUtil.

        Args:
            model (nn.Module): The model.
            optimizer (Optimizer): The optimizer.
            lr_scheduler (Any): The learning rate scheduler.
            scaler: (Any): The amp grad scaler.
            save_dir (str): The directory to save checkpoint.
            pretrain_model (str): Path of pretrain model.
            resume (bool): Whether to resume.
            to_cuda (bool, optional): Whether to remap storage locations to cuda. Read
                more detail at `torch.load.map_location`
                Default=`False`.
            ckpt2model_json (str or None): If set, checkpoint key will be remapped.
                Default=`None`.
            drop_model_params (bool): If set true, will use ckpt_key of
                ckpt2model_json to load model params,
                other parameters of the model will be discarded.
            trace_mode (bool): Whether to enable trace mode. Default=`False`.
        """
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._scaler = scaler
        self._save_dir = save_dir
        self._pretrain_model = pretrain_model
        self._enable_load_optimizer = enable_load_optimizer
        self._enable_load_lr_scheduler = enable_load_lr_scheduler
        self._checkpoint = {}
        self._resume = resume
        self._ckpt2model_json = ckpt2model_json
        self._drop_model_params = drop_model_params
        self._trace_mode = trace_mode
        self._use_interval = use_interval
        self._deque: deque = deque()
        if save_dir is not None:
            self._last_file = os.path.join(
                save_dir,
                TRAIN_CHECKPOINT_FORMAT.format(
                    get_pipeline_model_parallel_rank(), "last"
                ),
            )
            self._best_file = os.path.join(
                save_dir,
                TRAIN_CHECKPOINT_FORMAT.format(
                    get_pipeline_model_parallel_rank(), "best"
                ),
            )

        if self._resume:
            logger.warning(
                "Resume training, load last checkpoint: `%s`.", self._last_file
            )
            self._checkpoint = load_checkpoint(
                ckpt_path=self._last_file,
                to_cuda=to_cuda,
            )
        elif pretrain_model is not None:

            if isinstance(pretrain_model, list):
                logger.warning("=========> Use Pretrain Model List: %s", pretrain_model)
                _pretrain_model: str = pretrain_model[
                    get_pipeline_model_parallel_rank()
                ]
                logger.warning(
                    "Model Rank: %s, Load pretrain checkpoint: `%s`.",
                    get_pipeline_model_parallel_rank(),
                    _pretrain_model,
                )
            else:
                _pretrain_model = pretrain_model
                logger.warning("Load pretrain checkpoint: `%s`.", _pretrain_model)
            self._checkpoint = load_checkpoint(
                ckpt_path=_pretrain_model,
                to_cuda=to_cuda,
            )
        else:
            logger.warning("No pretrain model find.")

    @rank_zero_only
    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        max_save_ckpt_num: Optional[int] = None,
        save_by_freq: bool = False,
        save_best: bool = False,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch (int): Epoch id.
            step (int): Step id.
            save_by_freq (bool): If `True`, denotes checkpoint will be saved
            by the setted save frequency。
            save_best (bool): If `True`, checkpoint name will be `...checkpoint-best`
                and overwrite previous file.
                Default=`False`.
        """
        if self._save_dir is None:
            raise NotADirectoryError()

        state = {
            "epoch": epoch,
            "step": step,
            "model": self._model.state_dict(),
        }

        if self._optimizer is not None:
            state["optimizer"] = self._optimizer.state_dict()
        if self._lr_scheduler is not None:
            state["lr_scheduler"] = self._lr_scheduler.state_dict()
        if self._scaler is not None:
            state["scaler"] = self._scaler.state_dict()

        if not self._use_interval:
            logger.warning("Saving checkpoint, do not exit.")
            torch.save(state, self._last_file)
            logger.info("Save last model checkpoint: `%s`.", self._last_file)

        if save_by_freq:
            if self._use_interval:
                logger.warning("Saving checkpoint, do not exit.")
                torch.save(state, self._last_file)
                logger.info("Save last model checkpoint: `%s`.", self._last_file)

            epoch_or_step = f"epoch{epoch:0>3}_step{step:0>4}"
            ckpt_file_path = os.path.join(
                self._save_dir,
                TRAIN_CHECKPOINT_FORMAT.format(
                    get_pipeline_model_parallel_rank(), epoch_or_step
                ),
            )
            shutil.copyfile(self._last_file, ckpt_file_path)

            if max_save_ckpt_num is not None:
                if len(self._deque) >= max_save_ckpt_num:
                    del_ckpt_path = self._deque.popleft()
                    os.remove(del_ckpt_path)
                self._deque.append(ckpt_file_path)
            logger.info("Save model checkpoint: `%s`", ckpt_file_path)

        if save_best:
            shutil.copyfile(self._last_file, self._best_file)
            logger.info("Save best model checkpoint: `%s`.", self._best_file)

        logger.warning("Saving checkpoint done.")

    def _mapping_ckpt2model(
        self,
        model_state_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Map checkpoint key to model key.

        Args:
            model_state_dict (Dict[str, np.ndarray]): State dict of model.


        Returns:
            Remapped model_state_dict.
        """
        logger.info("Mapping ckpt keys to model keys by %s.", self._ckpt2model_json)

        # Read json.
        if self._ckpt2model_json is None:
            raise ValueError("self._ckpt2model_json should not be None.")

        with open(self._ckpt2model_json, "r", encoding="utf8") as json_file:
            ckpt2model_mapping = json.load(json_file)

        _model_state_dict = {}
        for ckpt_key, model_key in ckpt2model_mapping.items():
            if ckpt_key not in model_state_dict:
                raise KeyError(
                    f"ckpt_key not in "
                    f"model_state_dict.keys()={model_state_dict.keys()}."
                )
            _model_state_dict[model_key] = model_state_dict.pop(ckpt_key)
        if model_state_dict:
            if self._drop_model_params:
                logger.warning(
                    "Load only ckpt2model_json parameters"
                    "the original model parameters will be discarded"
                )
            else:
                _model_state_dict.update(**model_state_dict)
        model_state_dict = _model_state_dict
        return model_state_dict

    def load_model(self) -> None:
        """Load model."""
        if self._resume:
            # Resume.
            try:
                model_state_dict = self._checkpoint["model"]
                self._model.load_state_dict(
                    model_state_dict,
                    strict=True,
                )
                return
            except RuntimeError as runtime_error:
                error_msg = runtime_error.args[0]
                if "mismatch" in error_msg:
                    error_msg += (
                        "\nMeet mismatch error. Your model does not match checkpoint. "
                        "Set `resume`=`False`."
                    )
                raise RuntimeError(error_msg) from runtime_error
        else:
            if self._pretrain_model is None:
                logger.warning("Training from scratch. Skip load model.")
                return

            if "model" not in self._checkpoint:
                logger.warning(
                    "Key `model` not in checkpoint, Will treat checkpoint as "
                    "model_state_dict. checkpoint_keys()=%s.\n",
                    str(self._checkpoint.keys()),
                )
                model_state_dict = self._checkpoint
            else:
                model_state_dict = self._checkpoint["model"]

            # Modify state dict.
            if self._trace_mode:
                # Trace.
                logger.warning("Load ckpt on tracing mode.")
                for weight_key in list(model_state_dict.keys()):
                    if weight_key.startswith("module."):
                        new_weight_key = weight_key[7:]  # Remove `module.` prefix.
                        model_state_dict[new_weight_key] = model_state_dict.pop(
                            weight_key
                        )
            elif self._ckpt2model_json is not None:
                # If set checkpoint to model key mapping.
                if not self._ckpt2model_json.endswith("json"):
                    raise ValueError(
                        f"Expect `ckpt2model_json` is a path to json file but "
                        f"got {self._ckpt2model_json}."
                    )
                model_state_dict = self._mapping_ckpt2model(model_state_dict)

            # miss_key means key in model but not in checkpoint.
            # unexpect_key means key in checkpoint but not in model.
            # logger.info(model_state_dict.keys())
            miss_key, unexpect_key = self._model.load_state_dict(
                model_state_dict,
                strict=False,
            )

            logger.warning("Miss keys: %s.", str(miss_key))
            logger.warning("Unexpect keys: %s", str(unexpect_key))
            logger.warning("Load model done.")

    def load_scaler(self) -> None:
        """Load scaler."""
        if self._scaler is None:
            logger.warning("Scaler is None, skip load scaler.")
            return

        if self._resume:
            logger.warning("Load scaler.")
            self._scaler.load_state_dict(self._checkpoint["scaler"])
            logger.warning("Load scaler done.")
        else:
            if self._pretrain_model is None:
                logger.warning("Training from scratch. Skip load scaler.")
                return

            if "scaler" not in self._checkpoint:
                logger.warning(
                    "Key `scaler` not in checkpoint, skip load scaler. "
                    "checkpoint_keys()=%s.",
                    str(self._checkpoint.keys()),
                )
                return

            logger.warning("Load scaler.")
            self._scaler.load_state_dict(self._checkpoint["scaler"])
            logger.warning("Load scaler done.")

    def load_optimizer(self) -> None:
        """Load optimizer."""
        if self._optimizer is None:
            logger.warning("Optimizer is None, skip load optimizer.")
            return

        if self._resume:
            logger.warning("Load optimizer.")
            self._optimizer.load_state_dict(self._checkpoint["optimizer"])
            logger.warning("Load optimizer done.")
        # load weight & optimizer
        elif self._enable_load_optimizer:
            if self._pretrain_model is None:
                logger.warning("Training from scratch. Skip load optimizer.")
                return

            if "optimizer" not in self._checkpoint:
                logger.warning(
                    "Key `optimizer` not in checkpoint, skip load optimizer. "
                    "checkpoint_keys()=%s.",
                    str(self._checkpoint.keys()),
                )
                return

            logger.warning("Load optimizer.")
            self._optimizer.load_state_dict(self._checkpoint["optimizer"])
            logger.warning("Load optimizer done.")
        # only load model weight
        else:
            logger.warning("load_optimizer=False ,skip load optimizer.")
            return

    def load_lr_scheduler(self) -> None:
        """Load learning rate scheduler."""
        if self._lr_scheduler is None:
            logger.warning("lr_scheduler is None, skip load lr_scheduler.")
            return

        if self._resume:
            logger.warning("Load lr_scheduler.")
            self._lr_scheduler.load_state_dict(self._checkpoint["lr_scheduler"])
            logger.warning("Load lr_scheduler done.")
        # load weight & lr_scheduler
        elif self._enable_load_lr_scheduler:
            if self._pretrain_model is None:
                logger.warning("Training from scratch. Skip load lr_scheduler.")
                return

            if "lr_scheduler" not in self._checkpoint:
                logger.warning(
                    "Key `lr_scheduler` not in checkpoint, skip load lr_scheduler. "
                    "checkpoint_keys()=%s.",
                    str(self._checkpoint.keys()),
                )
                return

            logger.warning("Load lr_scheduler.")
            self._lr_scheduler.load_state_dict(self._checkpoint["lr_scheduler"])
            logger.warning("Load lr_scheduler done.")
        # only load model weight
        else:
            logger.warning("load_lr_scheduler=False ,skip load lr_scheduler.")
            return

    def load_step(self) -> int:
        """Load step."""
        if self._resume:
            logger.warning("Load step.")
            pre_step = self._checkpoint["step"]
            logger.warning("Load step done.")
        else:
            logger.warning("Not resume, train from step 0.")
            pre_step = 0
        return pre_step

    def load_epoch(self) -> int:
        """Load epoch."""
        if self._resume:
            logger.warning("Load epoch.")
            pre_epoch = self._checkpoint["epoch"]
            logger.warning("Load epoch done.")
        else:
            logger.warning("Not resume, train from epoch 0.")
            pre_epoch = 0
        return pre_epoch

    @property
    def checkpoint(self):
        """Return checkpoint."""
        return self._checkpoint

# -*- coding: utf-8 -*-
"""UniRobot tensorboard manager."""

from typing import Any
from typing import List

from torch.utils.tensorboard import SummaryWriter


class TensorboardUtil:
    """Tensorboard Manager."""

    def __init__(
        self,
        log_dir: str = "./",
        comment: str = "",
        filename_suffix: str = "",
        max_queue: int = 10,
        flush_secs: int = 120,
        rank: int = 0,
    ):
        """Init TensorboardManager Params.

        Args:
          log_dir (str): Save directory location.
          comment (str): Comment log_dir suffix appended to the default log_dir.
            For example, log_dir/xxxx.[comment]
          filename_suffix (str): Suffix added to all event filenames \
            in the log_dir directory.  For example, log_dir/xxxx.[comment].suffix
          max_queue (int): Size of the queue for pending events and summaries \
            before one of the ‘add’ calls forces a flush to disk. \
            Default: 10.
          flush_secs (int): How often, in seconds, \
            to flush the pending events and summaries to disk.\
            Default: 120 s.
        """
        filename_suffix = filename_suffix + f"_rank_{rank}"
        self.rank = rank
        self._writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

    def log_grad(self, model: Any, global_step: int = 0):
        """Log Model Parameters/Gradient.

        Args:
            model (nn.Module): A nn.Module instance.
            global_step_idx (int): The index of global step.
        """
        for tag, value in model.named_parameters():
            if value.grad is not None:
                self._writer.add_histogram("grad/" + tag, value.grad.cpu(), global_step)
                self._writer.add_histogram("value/" + tag, value.cpu(), global_step)

    def log_scaler(self, value_list_per_step: List, global_step_idx: int = 0):
        """Log scaler value.

        Args:
          value_list_per_step (list): ["name: value",...]
          global_step_idx (int): The index of global step.
        """
        # process value_list_per_step as dict:
        for str_item in value_list_per_step:
            tag = str_item.split(":")[0].strip()
            value = str_item.split(":")[1].strip()
            if tag == "Task":
                continue
            if ("lr" in tag) and ("cost" not in tag):
                tag = "learning-rate/" + tag
            if ("loss" in tag) and ("cost" not in tag):
                tag = "loss-info/" + tag
            if (
                ("cost" in tag)
                or ("iter" in tag)
                or (("sample" in tag) and ("lr" not in tag))
            ):
                tag = "efficiency/" + tag
            self._writer.add_scalar(tag, float(value), global_step=global_step_idx)

    def close(self):
        """Close Writer."""
        self._writer.close()

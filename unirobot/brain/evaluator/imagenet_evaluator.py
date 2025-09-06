# -*- coding: utf-8 -*-
"""Imagenet Evaluator."""

import logging
from typing import Any
from typing import Dict

import torch
import torch.distributed as dist

from unirobot.brain.evaluator.base_evaluator import BaseEvaluator
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_group,
)


logger = logging.getLogger(__name__)


class ImageNetEvaluator(BaseEvaluator):
    """ImageNet Evaluator.

    Args:
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(
        self,
        use_gpu: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__(use_gpu=use_gpu)
        self._hit_count = torch.Tensor([0])
        self._sample_count = torch.Tensor([0])

        if self._use_gpu:
            self._hit_count = self._hit_count.cuda(torch.cuda.current_device())
            self._sample_count = self._sample_count.cuda(torch.cuda.current_device())

        self._results: Dict[str, Any] = {}

    def set_zero(self):
        """Set Init evaluator."""
        self._hit_count = torch.Tensor([0])
        self._sample_count = torch.Tensor([0])
        if self._use_gpu:
            self._hit_count = self._hit_count.cuda(torch.cuda.current_device())
            self._sample_count = self._sample_count.cuda(torch.cuda.current_device())
        self._results: Dict[str, Any] = {}

    def eval(
        self,
        model_outputs: Dict[str, Any],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement of Evaluator.

        Args:
            model_outputs (Tuple[Any, ...]): Model_outputs: [BS, ClassNum].
            model_inputs (Dict[str, Any]): gt: [BS] value[0 - (ClassNum - 1)].

        Returns:
            Evaluation result.
        """
        batch_size = model_outputs["model_outputs"].size(0)
        max_value_index = torch.argmax(model_outputs["model_outputs"], dim=1)
        hit_num = torch.sum(max_value_index == model_inputs["gt"])

        self._sample_count = self._sample_count + batch_size
        self._hit_count = self._hit_count + hit_num
        self._results["hit_count"] = self._hit_count
        self._results["sample_count"] = self._sample_count
        self._results["top1"] = self._hit_count / self._sample_count
        return self._results

    def get_specific_result(
        self,
        query_key: str,
    ) -> Dict[str, Any]:
        """Return evaluation result by query key.

        Args:
            query_key (str): Key of evaluation result.

        Returns:
            Specific result of query key.
        """
        return self._results[query_key]

    def dist_all_reduce(
        self,
        eval_results: Dict[Any, Any],
    ) -> Dict[Any, Any]:
        """All reduce by torch.distributed as dist.

        Args:
            eval_results (Dict[Any, Any]): Eval results of all epoch.

        Returns:
            eval_results.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(eval_results["hit_count"], group=get_data_parallel_group())
            dist.all_reduce(
                eval_results["sample_count"], group=get_data_parallel_group()
            )
            dist.all_reduce(
                eval_results["top1"].div_(
                    dist.get_world_size(group=get_data_parallel_group())
                ),
                group=get_data_parallel_group(),
            )

        # reset start_step
        eval_results["hit_count"] = eval_results["hit_count"].item()
        eval_results["sample_count"] = eval_results["sample_count"].item()
        eval_results["top1"] = eval_results["top1"].item()

        return eval_results

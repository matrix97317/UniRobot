# -*- coding: utf-8 -*-
"""UniRobot BaseEvaluator."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict


class BaseEvaluator(ABC):
    """Evaluator Abstract Interface.

    Args:
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize."""
        self._use_gpu = use_gpu

    @abstractmethod
    def eval(
        self,
        model_outputs: Any,
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement of Evaluator.

        Args:
            model_outputs (Tuple[Any, ...]): Model_outputs.
            model_inputs (Dict[str, Any]): Model_inputs, i.e. ground truth.

        Returns:
            Evaluation result.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @abstractmethod
    def dist_all_reduce(
        self,
        eval_results: Dict[Any, Any],
    ) -> Dict[Any, Any]:
        """All reduce by torch.distributed as dist.

        See example at ::class::ImageNetEvaluator.

        Args:
            eval_results (Dict[Any, Any]): Eval results of all epoch.

        Returns:
            eval_results.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_zero(self):
        """Set Init evaluator."""
        raise NotImplementedError()

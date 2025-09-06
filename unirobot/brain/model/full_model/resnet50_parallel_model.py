# -*- coding: utf-8 -*-
"""demo model.
"""
from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Tuple

from torch import Tensor

from unirobot.brain.model.full_model.base_full_model import BaseFullModel
from unirobot.utils.unirobot_slot import ENCODER


class ParallelRes50(BaseFullModel):
    """Define Task Model.

    Args:
        sub_module_cfg (Dict[str, Any]): Config dict of sub module.
        train_mode (bool): Whether to enable training. Default=`True`.
    """

    def __init__(
        self,
        sub_module_cfg: Dict[str, Any],
        train_mode: bool = True,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """Init Task Model base sub components."""
        super().__init__(
            sub_module_cfg=sub_module_cfg,
            train_mode=train_mode,
            **kwargs,
        )
        self._backbone = ENCODER.build(sub_module_cfg)
        self._batch_size = batch_size
        self._batch_data = None

    def init_weight(self) -> None:
        """Init model weight."""
        self._backbone.init_weight()

    def model_tensor_shape_list(self):
        """Return out tensor shape."""
        # return [
        #     [
        #         (self._batch_size, 256, 56, 56),
        #         (self._batch_size, 512, 28, 28),
        #         (self._batch_size, 1024, 14, 14),
        #         (self._batch_size, 2048, 7, 7),
        #     ],
        #     (self._batch_size, 1000),
        # ]
        return [
            (self._batch_size, 256, 56, 56),
            (self._batch_size, 512, 28, 28),
            (self._batch_size, 1024, 14, 14),
            (self._batch_size, 1000),
        ]

    def set_batch_data(self, batch_data):
        """Set batch data."""
        self._batch_data = batch_data

    def get_batch_data(self):
        """Return batch data."""
        return self._batch_data

    def parse_inputs_data(self, inputs_data):  # pylint: disable=no-self-use
        """Parse inputs data."""
        if isinstance(inputs_data, dict):
            return [inputs_data["image"]]
        return [inputs_data]

    def wrap_outputs_data(self, outputs_data):  # pylint: disable=no-self-use
        """Wrap outputs data."""
        return {"model_outputs": outputs_data[0]}

    def infer_forward(self, image: Tensor) -> Tuple[Any, ...]:
        """Infer forward.

        Args:
            image (Tensor): Image Tensor.

        Returns:
            Outputs of infer, Union[Dict[str, Any], torch.Tensor].
        """
        return namedtuple("model_outputs", ["model_outputs"])(self._backbone(image))

    def train_forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train forward.

        Args:
            inputs (Dict[str, Any]): Inputs of infer.

        Returns:
            Outputs of train, Dict[str, Any].
        """
        return self._backbone(inputs)


class ParallelRes50V2(ParallelRes50):
    """Define Task Model."""

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)

    def model_tensor_shape_list(self):
        """Return out tensor shape."""
        return [
            [
                (self._batch_size, 256, 56, 56),
                (self._batch_size, 512, 28, 28),
                (self._batch_size, 1024, 14, 14),
                (self._batch_size, 2048, 7, 7),
            ],
            (self._batch_size, 1000),
        ]


class ParallelRes50V3(ParallelRes50):
    """Define Task Model."""

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)

    def model_tensor_shape_list(self):
        """Return out tensor shape."""
        return [
            (self._batch_size, 1000),
        ]

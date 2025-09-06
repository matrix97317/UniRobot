# -*- coding: utf-8 -*-
"""UniRobot ModelFlow."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

import torch
from torch import nn

from unirobot.brain.model.full_model.base_full_model import BaseFullModel
from unirobot.utils.unirobot_slot import FULL_MODEL
from unirobot.utils.unirobot_slot import LOSS


class ModelFlow(torch.nn.Module):
    """Interface of model flow.

    Args:
        full_model_cfg (Dict[str, Any]): Config dict of full model.
        loss_func_cfg (Dict[str, Any]): Config dict of loss function. Default=`None`.
        postprocess_cfg (Dict[str, Any]): Config dict of postprocess.
        train_mode (bool): Whether to enable training mode. Default=`True`.
        enable_pbn (bool): Whether to enable pbn. Default=`False`.
    """

    def __init__(
        self,
        full_model_cfg: Dict[str, Any],
        loss_func_cfg: Optional[Dict[str, Any]] = None,
        postprocess_cfg: Optional[Dict[str, Any]] = None,
        train_mode: bool = True,
        enable_pbn: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the basic components of full model."""
        super().__init__(**kwargs)

        self._full_model = FULL_MODEL.build(full_model_cfg)
        if not isinstance(self._full_model, BaseFullModel):
            raise ValueError(
                f"Expect type(self._full_model)=BaseFullModel, "
                f"but got {type(self._full_model)}."
            )

        self._loss_func: Union[nn.ModuleDict, Callable, None] = None
        if train_mode:
            # Note: Implement build losses is needed if enable train mode.
            if loss_func_cfg is None:
                raise ValueError(
                    "If enable train mode, expect loss_func_cfg is not None, "
                    "but got None"
                )
            self._loss_func = LOSS.build(loss_func_cfg)

        # Note: Implement build postprocess is needed if postprocess cfg is not None.
        if postprocess_cfg is None:
            self._postprocess_layer = None

        self._full_model.set_mode(train_mode)
        self._train_mode = train_mode
        self._enable_pbn = enable_pbn

    def trace_model_inputs_hook(self, model_inputs: Any) -> Any:
        """Convert DataLoader's Data as the inputs of trace model."""
        raise NotImplementedError()

    def trace_model_outputs_hook(self, model_outputs: Any) -> Any:
        """Convert the outputs of trace model as Evaluator's inputs."""
        raise NotImplementedError()

    @staticmethod
    def load_traced_model(ckpt_path) -> Any:
        """Load traced model."""
        traced_model = torch.jit.load(
            ckpt_path, map_location=torch.device(torch.cuda.current_device())
        )
        traced_model = traced_model.cuda(device=torch.cuda.current_device())

        return traced_model

    def forward(
        self,
        inputs: Dict[str, Any],
    ) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], Tuple[Any, ...]]:
        """Forward.

        Args:
            inputs (Dict[str, Any]): Model inputs.

        Returns:
            If `train_mode`=True, return model train outputs and loss_dict.
            If `train_mode`=False, return model infer outputs.
        """
        if not self._train_mode:
            # NOTE: For `torch.jit.trace()`, the arguments should be several Tensors,
            # so `**inputs` instead of `inputs` (a dict) itself is used here.
            infer_outputs: Tuple[Any, ...] = self._full_model(**inputs)
            if self._postprocess_layer:
                infer_outputs = self._postprocess_layer(infer_outputs)
            return infer_outputs

        train_outputs = self._full_model(inputs)
        if self._postprocess_layer:
            train_outputs = self._postprocess_layer(train_outputs)
        loss_dict = self.compute_loss(train_outputs, inputs)
        return train_outputs, loss_dict

    def compute_loss(
        self,
        inputs: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute task model loss.

        Args:
            inputs (Dict[str, Any]): Module inputs.
            targets (Dict[str, Any]): Ground truth.

        Returns:
            Loss results.
        """
        if self._loss_func is not None and callable(self._loss_func):
            return self._loss_func(inputs, targets)
        raise NotImplementedError()

    def get_trace_model(self) -> Any:
        """Return trace model."""
        example_forward_input = torch.rand(1, 3, 224, 224)

        return torch.jit.trace(self, example_forward_input)

    def get_optim_policies(
        self,
    ) -> List[Dict[str, Any]]:
        """Get optimizer policies.

        Returns:
            Return optim policies for different model layer.
        """
        return [
            {
                "params": list(self.parameters()),
                "lr_mult": 1,
                "decay_mult": 1,
                "name": "all_params",
            },
        ]

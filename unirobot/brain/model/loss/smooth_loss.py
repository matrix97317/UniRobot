# -*- coding: utf-8 -*-
"""LabelSmoothingLoss."""

from typing import Any
from typing import Dict

import torch

from unirobot.brain.model.loss.base_loss import BaseLoss


class LabelSmoothLoss(BaseLoss):
    """CrossEntropyLoss."""

    def __init__(
        self,
        name: str = "label_smooth_loss",
        smooth_value: float = 0.1,
    ) -> None:
        """Initialize."""
        super().__init__(name=name)
        self._confidence = 1.0 - smooth_value
        self._smoothing = smooth_value

    def forward(  # pylint: disable=no-self-use
        self,
        model_outputs: Dict[str, Any],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Loss."""
        # pred_logits shape: [BS, N]
        # label shape: [N]
        # normalize and log
        pred_logits, label = model_outputs["model_outputs"], model_inputs["gt"]
        # print("pred_logits", pred_logits.shape)
        # print("label", label.shape)
        logprobs = torch.nn.functional.log_softmax(pred_logits, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=label.long().unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self._confidence * nll_loss + self._smoothing * smooth_loss
        return {"loss_cls": loss.mean()}



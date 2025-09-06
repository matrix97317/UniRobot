# -*- coding: utf-8 -*-
"""CrossEntropyLoss."""

from typing import Any
from typing import Dict

import torch

from unirobot.brain.model.loss.base_loss import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """CrossEntropyLoss."""

    def __init__(
        self,
        name: str = "cross_entropy_loss",
    ) -> None:
        """Initialize."""
        super().__init__(name=name)

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
        pred_logits_norm_log = torch.log(torch.softmax(pred_logits, dim=1))
        loss = torch.nn.NLLLoss()(pred_logits_norm_log, label.long())

        return {"loss_cls": loss}

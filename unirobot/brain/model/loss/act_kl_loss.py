# -*- coding: utf-8 -*-
"""LabelSmoothingLoss."""

from typing import Any
from typing import Dict

import torch
from torch.nn import functional as F

from unirobot.brain.model.loss.base_loss import BaseLoss


class ACTKLLoss(BaseLoss):
    """CrossEntropyLoss."""

    def __init__(
        self,
        name: str = "act_kl_loss",
        kl_weight: float = 10,
    ) -> None:
        """Initialize."""
        super().__init__(name=name)
        self.kl_weight = kl_weight

    def forward(  # pylint: disable=no-self-use
        self,
        model_outputs: Dict[str, Any],
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Loss."""
        # pred_logits shape: [BS, N]
        # label shape: [N]
        # normalize and log
        actions = model_outputs["actions"]
        is_pad = model_outputs["is_pad"]

        a_hat = model_outputs["a_hat"]
        mu, logvar = model_outputs["mu"], model_outputs["logvar"]

        total_kld, dim_wise_kld, mean_kld = self._kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]
        loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
        return loss_dict

    def _kl_divergence(self, mu, logvar):
        """KL divergence loss."""
        batch_size = mu.size(0)
        if batch_size == 0:
            raise ValueError("batch_size == 0")
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld

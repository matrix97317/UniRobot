# -*- coding: utf-8 -*-
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron optimizer."""
# pylint: skip-file
# flake8: noqa
import logging
from abc import ABC
from abc import abstractmethod

import torch

from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_model_parallel_group,
)
from unirobot.brain.infra.distributed.optimizer.clip_grads import clip_grad_norm_fp32
from unirobot.brain.infra.distributed.optimizer.clip_grads import count_zeros_fp32


logger = logging.getLogger(__name__)


def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


class MegatronOptimizer(ABC):
    """Megatron Optimizer."""

    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
        ddp_type,
        models,
    ):
        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        if not self.optimizer:
            logger.error("no optimizer is provided.")
        # Set gradient clipping and logging params.
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp
        self.ddp_type = ddp_type
        # 'models' are retained for access to the contiguous grad buffers.
        # (see distributed optimizer)
        self.models = models

        if self.use_contiguous_buffers_in_local_ddp:
            if not self.params_have_main_grad:
                logger.error(
                    "use of contiguous buffer requires that params have main grad"
                )

    def get_parameters(self):
        """Return parameters."""
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def get_main_grads_for_grad_norm(self):
        """Return grads for norm."""
        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = True
            is_not_tp_duplicate = True
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def get_model_parallel_group(self):
        """Return model parallel group."""
        return get_model_parallel_group()

    def clip_grad_norm(self, clip_grad):
        """Clip grad norm."""
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params,
            grads_for_norm,
            clip_grad,
            model_parallel_group=self.get_model_parallel_group(),
        )

    def count_zeros(self):
        """Count grad zeros."""
        params = self.get_parameters()
        return count_zeros_fp32(
            params, model_parallel_group=self.get_model_parallel_group()
        )

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        """Set grad zero."""
        pass

    @abstractmethod
    def get_loss_scale(self):
        """Returan loss scale value."""
        pass

    def scale_loss(self, loss):
        """Scaling loss."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def reload_model_params(self):
        """Reload model params."""
        pass

    @abstractmethod
    def state_dict(self):
        """Return state dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load state dict."""
        pass

    # Promote state so it can be retrieved or set via
    # "optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    @abstractmethod
    def step(self):
        """Update model params."""
        pass

    def gather_model_params(self):
        """Gather model params."""
        pass

    def reduce_model_grads(self):
        """All-reduce all grads, and all-reduce embeddings."""
        # All-reduce if needed.
        if self.ddp_type == "local":
            for model in self.models:
                model.allreduce_gradients()


class FP32Optimizer(MegatronOptimizer):
    """Fp32 Optimizer."""

    def __init__(
        self,
        optimizer,
        clip_grad,
        log_num_zeros_in_grad,
        params_have_main_grad,
        use_contiguous_buffers_in_local_ddp,
        ddp_type,
        models,
        scaler=None,
        use_fsdp=False,
    ):
        """Init."""
        super(FP32Optimizer, self).__init__(
            optimizer,
            clip_grad,
            log_num_zeros_in_grad,
            params_have_main_grad,
            use_contiguous_buffers_in_local_ddp,
            ddp_type,
            models,
        )
        self._scale = torch.cuda.FloatTensor([1.0])
        self._amp_scaler = scaler
        self._use_fsdp = use_fsdp

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer."""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group["params"], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def step(self):
        """Update model params."""
        if not self._use_fsdp:
            # Copy main_grads to grads.
            if self.params_have_main_grad:
                for param_group in self.optimizer.param_groups:
                    for param in param_group["params"]:
                        param.grad = param.main_grad

                        # Safe to de-reference model's main_grad after copying.
                        # (If using contiguous buffers, main_grad's memory should
                        # persist and therefore should not be deallocated.)
                        if not self.use_contiguous_buffers_in_local_ddp:
                            param.main_grad = None

        # Clip gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            if self._amp_scaler is not None:
                self._amp_scaler.unscale_(self.optimizer)
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if self.log_num_zeros_in_grad else None
        # Update parameters.
        if self._amp_scaler is not None:
            # self.optimizer.step()
            self._amp_scaler.step(self.optimizer)
            self._amp_scaler.update()
        else:
            self.optimizer.step()

        # No overflow for FP32 optimizer.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        """Reload model params."""
        pass

    def state_dict(self):
        """Return optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict to optimizer."""
        self.optimizer.load_state_dict(state_dict)

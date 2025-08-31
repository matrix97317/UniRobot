# -*- coding: utf-8 -*-
"""Pipeline Scheme."""
# pylint: skip-file
# flake8: noqa
import logging
from contextlib import contextmanager

import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

import unirobot.brain.infra.distributed.communication.p2p_com as p2p_communication
from unirobot.brain.infra.distributed.data_parallel.wrap_model import (
    DistributedDataParallel as LocalDDP,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_group,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_world_size,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    is_pipeline_first_stage,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    is_pipeline_last_stage,
)
from unirobot.brain.infra.distributed.utils import unwrap_model


logger = logging.getLogger(__name__)


def _train_parse_losses(
    losses,
    _enable_sync_loss=True,
):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.
            It mainly supports two forms.
            1. {'loss_name': [l_1, l_2, ...,l_bs_size](type: torch.Tensor)}
            2. {'loss_name': [[l_1, l_2, ...,l_bs_size],...](type: list)}.
        task_name (str): Indicate to parse losses for the specific task,
            defaults to `""` if no task specified.

    Returns:
        Tensor: loss, loss is the loss tensor which may be a weighted sum of all
        losses.
    """
    local_loss_vars = {}

    # BatchSize mean.
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            local_loss_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            local_loss_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f"{loss_name} is not a tensor or list of tensors")

    # Task sum.
    loss = sum(_value for _key, _value in local_loss_vars.items() if "loss" in _key)

    local_loss_vars["loss"] = loss

    # Device mean.
    for loss_name, loss_value in local_loss_vars.items():
        # reduce loss when distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            loss_value = loss_value.data.clone()
            if _enable_sync_loss:
                torch.distributed.all_reduce(
                    loss_value.div_(
                        torch.distributed.get_world_size(
                            group=get_data_parallel_group()
                        )
                    ),
                    group=get_data_parallel_group(),
                )
        local_loss_vars[loss_name] = loss_value

    return loss, local_loss_vars


def deallocate_output_tensor(out):
    """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    """
    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise ValueError(f"expected Tensor, found {type(out).__name__}")
    out.data = torch.empty(
        (1,),
        device=out.device,
        dtype=out.dtype,
    )


def custom_backward(output, grad_output):
    """Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    """

    if output.numel() != 1:
        raise ValueError(
            "output should be pseudo-'freed' in schedule, to optimize memory"
        )
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"output == ' {type(output).__name__} '.")
    if not isinstance(grad_output, (torch.Tensor, type(None))):
        raise ValueError(f"grad_output == ' {type(grad_output).__name__}'.")

    # Handle scalar output
    if grad_output is None:
        if output.numel() != 1:
            raise ValueError("implicit grad requires scalar output.")
        grad_output = torch.ones_like(
            output,
            memory_format=torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    data_loader,
    loss_func,
    model,
    input_tensor,
    forward_data_store,
    collect_non_loss_data=False,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""

    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP))
    num_microbatches = get_pipeline_model_parallel_world_size()
    # print("num_microbatche", num_microbatches)
    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    batch_data = data_loader.get_batch_data()
    unwrapped_model.set_batch_data(batch_data)

    # logger.warning(
    #     "rank: {} gt: {}".format(torch.distributed.get_rank(), batch_data["gt"])
    # )
    if is_pipeline_first_stage():
        input_tensor = unwrapped_model.parse_inputs_data(batch_data)

        output_tensor = model(input_tensor)
    else:
        output_tensor = model(input_tensor[0])

    if is_pipeline_last_stage():

        batch_data = unwrapped_model.get_batch_data()
        wrap_tensor = unwrapped_model.wrap_outputs_data(output_tensor)

        if not collect_non_loss_data:
            losses = loss_func(wrap_tensor, batch_data)
            loss, local_loss_vars = _train_parse_losses(losses)
            output_tensor = loss / num_microbatches
            forward_data_store.append(local_loss_vars)
        else:
            forward_data_store.append([wrap_tensor, batch_data])
        # logger.warning(
        #     "rank: {} gt: {}".format(torch.distributed.get_rank(), forward_data_store)
        # )

    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None:
        output_tensor = optimizer.scale_loss(output_tensor[0])
    custom_backward(output_tensor[0], output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    return input_tensor_grad


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(
    data_loader,
    loss_func,
    model,
    optimizer,
    forward_only=False,
    collect_non_loss_data=False,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    if len(model) != 1:
        raise ValueError(
            "`forward_backward_no_pipelining` just suport one chunk model."
        )
    num_microbatches = get_pipeline_model_parallel_world_size()
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(
                data_loader,
                loss_func,
                model,
                input_tensor,
                forward_data_store,
                collect_non_loss_data,
            )

            if not forward_only:
                backward_step(
                    optimizer, input_tensor, output_tensor, output_tensor_grad
                )

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        data_loader,
        loss_func,
        model,
        input_tensor,
        forward_data_store,
        collect_non_loss_data,
    )
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return forward_data_store


def get_out_tensor_shapes(rank, model_tensor_shape_list):
    """Return tensor shape by model_tensor_shape_list."""
    tensor_shapes = []
    if rank < 0 or rank == get_pipeline_model_parallel_world_size():
        return tensor_shapes
    else:
        tensor_shapes.append(model_tensor_shape_list[rank])
    return tensor_shapes


def recv_forward(tensor_shapes, cfg):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, cfg=cfg))
    return input_tensors


def recv_backward(tensor_shapes, cfg):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(
                p2p_communication.recv_backward(tensor_shape, cfg=cfg)
            )
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, cfg):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape, cfg=cfg)


def send_backward(input_tensor_grads, tensor_shapes, cfg):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape, cfg=cfg)


def send_forward_recv_backward(output_tensors, tensor_shapes, cfg):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, cfg=cfg
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, cfg):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, cfg=cfg
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_with_1F1B(
    data_loader,
    loss_func,
    model,
    optimizer,
    cfg,
    forward_only=False,
    collect_non_loss_data=False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    if len(model) != 1:
        raise ValueError(
            "`forward_backward_pipelining_with_1F1B` just suport one chunk model."
        )
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_pipeline_model_parallel_world_size()
    num_warmup_microbatches = (
        get_pipeline_model_parallel_world_size()
        - get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP))
    model_tensor_shape_list = unwrapped_model.model_tensor_shape_list()
    rank = get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_out_tensor_shapes(rank - 1, model_tensor_shape_list)
    send_tensor_shapes = get_out_tensor_shapes(rank, model_tensor_shape_list)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(recv_tensor_shapes, cfg=cfg)
        output_tensor = forward_step(
            data_loader,
            loss_func,
            model,
            input_tensor,
            forward_data_store,
            collect_non_loss_data,
        )
        send_forward(output_tensor, send_tensor_shapes, cfg=cfg)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, cfg=cfg)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        output_tensor = forward_step(
            data_loader,
            loss_func,
            model,
            input_tensor,
            forward_data_store,
            collect_non_loss_data,
        )
        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, cfg=cfg)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, cfg=cfg)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, cfg=cfg
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0])

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, cfg=cfg)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, cfg=cfg
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, cfg=cfg)

            input_tensor_grad = backward_step(
                optimizer, input_tensor, output_tensor, output_tensor_grad
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, cfg=cfg)
    # logger.warning(
    #     "rank: {} gt: {}".format(torch.distributed.get_rank(), len(forward_data_store))
    # )

    return forward_data_store

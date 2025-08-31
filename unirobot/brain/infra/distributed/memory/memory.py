# -*- coding: utf-8 -*-
"""Utility functions used throughout Megatron core."""
# pylint: skip-file
import logging
import operator
from functools import reduce

import torch


logger = logging.getLogger(__name__)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations."""

    def __init__(self):
        """Init."""
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        """Return tensor."""
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    out = torch.empty(
        (1,),
        dtype=inp.dtype,
        device=inp.device,
        requires_grad=requires_grad,
    )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    """
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    """

    @staticmethod
    def forward(ctx, inp, requires_grad):
        """Forward."""
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward."""
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    """
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    """
    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    """Assert that a tensor is not a view."""
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor._base is not None:
        logger.error(
            "Ensure tensor._base is None before setting tensor.data or storing "
            "tensor to memory buffer. Otherwise, a memory leak will occur (and "
            "likely accumulate over iterations). %s",
            extra_msg,
        )
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    """
    if tensor._base is None:
        extra_msg = "tensor._base is None."
    else:
        extra_msg = (
            f"tensor._base has shape {tensor._base.shape}, "
            "and new_data_tensor has shape {new_data_tensor.shape}."
        )
    assert_viewless_tensor(tensor, extra_msg=extra_msg)
    tensor.data = new_data_tensor


# A dictionary of all the memory buffers allocated.
_MEM_BUFFS = dict()


def allocate_mem_buff(name, numel, dtype, track_usage):
    """Allocate a memory buffer."""
    if name in _MEM_BUFFS:
        logger.error("memory buffer %s already allocated.", name)
    _MEM_BUFFS[name] = MemoryBuffer(name, numel, dtype, track_usage)
    return _MEM_BUFFS[name]


def get_mem_buff(name):
    """Get the memory buffer."""
    return _MEM_BUFFS[name]


class MemoryBuffer:
    """Contiguous memory buffer.

    Allocate a contiguous memory of type `dtype` and size `numel`. It is
    used to reduce memory fragmentation.

    Usage: After the allocation, the `_start` index is set tot the first
           index of the memory. A memory chunk starting from `_start` index
           can be `allocated` for an input tensor, with the elements of the
           tensor being coppied. The buffer can be reused by resetting the
           `_start` index.
    """

    def __init__(self, name, numel, dtype, track_usage):
        """Init."""
        if torch.distributed.get_rank() == 0:
            element_size = torch.tensor([], dtype=dtype).element_size()
            logger.info(
                "> building the %s memory buffer with %s num elements "
                "and %s dtype %s MB)...",
                name,
                numel,
                dtype,
                numel * element_size / 1024 / 1024,
            )

        self.name = name
        self.numel = numel
        self.dtype = dtype
        self.data = torch.empty(
            self.numel,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # Index tracking the start of the free memory.
        self._start = 0

        # Values used for tracking usage.
        self.track_usage = track_usage
        if self.track_usage:
            self.in_use_value = 0.0
            self.total_value = 0.0

    def reset(self):
        """Reset the buffer start index to the beginning of the buffer."""
        self._start = 0

    def is_in_use(self):
        """Whether the current buffer hold on to any memory."""
        return self._start > 0

    def numel_in_use(self):
        """Return number of elements in use."""
        return self._start

    def add(self, tensor):
        """Allocate a chunk of memory."""
        if tensor.dtype != self.dtype:
            logger.error(
                "Input tensor type %s different from buffer type %s",
                tensor.dtype,
                self.dtype,
            )
        # Number of elements of the input tensor.
        tensor_numel = torch.numel(tensor)
        new_start = self._start + tensor_numel
        if new_start > self.numel:
            logger.error(
                "Not enough memory left in the buffer (%s > %s)",
                tensor_numel,
                (self.numel - self._start),
            )
        # New tensor is a view into the memory.
        new_tensor = self.data[self._start : new_start]
        self._start = new_start
        new_tensor = new_tensor.view(tensor.shape)
        new_tensor.copy_(tensor)
        # Return a pointer to the new tensor.
        return new_tensor

    def get_data(self):
        """Return the data currently in use."""
        if self.track_usage:
            self.in_use_value += float(self._start)
            self.total_value += float(self.numel)
        return self.data[: self._start]

    def print_average_usage(self):
        """Print memory usage average over time."""
        if not self.track_usage:
            logger.error("You need to enable track usage.")
        if torch.distributed.get_rank() == 0:
            logger.info(
                " > usage of %s  memory buffer: %f %",
                self.name,
                (self.in_use_value * 100.0 / self.total_value),
            )


class RingMemBuffer:
    """A ring of memory buffers."""

    def __init__(self, name, num_buffers, numel, dtype, track_usage):
        """init."""
        self.num_buffers = num_buffers
        self.buffers = [
            allocate_mem_buff(name + f" {i}", numel, dtype, track_usage)
            for i in range(num_buffers)
        ]
        self._index = -1

    def get_next_buffer(self):
        """Return next buffer."""
        self._index += 1
        self._index = self._index % self.num_buffers
        buff = self.buffers[self._index]
        if buff.is_in_use():
            logger.error("buffer is already in use.")
        return buff

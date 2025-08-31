# -*- coding: utf-8 -*-
"""Model and data parallel groups."""
# pylint: skip-file
# mypy: ignore-errors
# flake8: noqa
import logging
from typing import Optional

import numpy as np
import torch

from unirobot.brain.infra.distributed.memory.memory import GlobalMemoryBuffer


logger = logging.getLogger(__name__)

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None


def generate_gpu_topo(pipeline_size, data_parallel_size, world_size, gpu_per_node):
    machine_groups = {}
    for idx, machine_group_start in enumerate(range(0, world_size, gpu_per_node)):
        # print("s", machine_group_start)
        machine_group = list(range(machine_group_start, gpu_per_node * (idx + 1), 1))
        if len(machine_group) % pipeline_size != 0:
            raise ValueError(
                "machine nums not be divided with no remainder. "
                f"machine_group size {len(machine_group)}, pipeline_size {pipeline_size}"
            )
        pipeline_group_size_per_machine = int(len(machine_group) / pipeline_size)
        machine_groups[idx] = []
        for pipeline_group_index in range(pipeline_group_size_per_machine):
            machine_groups[idx].append(
                machine_group[
                    pipeline_group_index
                    * pipeline_size : (pipeline_group_index + 1)
                    * pipeline_size
                ]
            )

    machine_nums = len(machine_groups.keys())
    machine_visited = {}
    pipeline_parallel_group = []
    for i in range(data_parallel_size):
        machine_index = i % machine_nums
        if machine_index not in machine_visited:
            machine_visited[machine_index] = 0
            pipeline_parallel_group.append(
                machine_groups[machine_index][machine_visited[machine_index]]
            )
        else:
            machine_visited[machine_index] += 1
            pipeline_parallel_group.append(
                machine_groups[machine_index][machine_visited[machine_index]]
            )
    data_parallel_group = np.transpose(np.array(pipeline_parallel_group)).tolist()
    return data_parallel_group, pipeline_parallel_group


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used
        for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs
        used for pipeline model parallelism.
        virtual_pipeline_model_parallel_size:
            number of virtual stages (interleaved pipeline).
        pipeline_model_parallel_split_rank:
            for models with both encoder and decoder,
            rank in pipeline with split point.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7],
            [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7],
            [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13],
            [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        logger.error("torch distributation not be initialized.")
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size "
            f"({pipeline_model_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size
    )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with "
                "interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size
        )

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    data_parallel_topo, pipeline_parallel_topo = generate_gpu_topo(
        pipeline_model_parallel_size, data_parallel_size, world_size, 8
    )
    # data_parallel_topo = [
    #     [0, 8, 16, 4, 12, 20],
    #     [1, 9, 17, 5, 13, 21],
    #     [2, 10, 18, 6, 14, 22],
    #     [3, 11, 19, 7, 15, 23],
    # ]
    # pipeline_parallel_topo = [
    #     [0, 1, 2, 3],
    #     [4, 5, 6, 7],
    #     [8, 9, 10, 11],
    #     [12, 13, 14, 15],
    #     [16, 17, 18, 19],
    #     [20, 21, 22, 23],
    # ]

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    if _DATA_PARALLEL_GROUP is not None:
        logger.error("data parallel group is already initialized")
    all_data_parallel_group_ranks = []

    for ranks in data_parallel_topo:
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            # group = torch.distributed.new_group(ranks)
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GLOBAL_RANKS = ranks
    all_data_parallel_group_ranks = data_parallel_topo

    # for i in range(pipeline_model_parallel_size):
    #     start_rank = i * num_pipeline_model_parallel_groups
    #     end_rank = (i + 1) * num_pipeline_model_parallel_groups
    #     for j in range(tensor_model_parallel_size):
    #         ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
    #         all_data_parallel_group_ranks.append(list(ranks))

    #         group = torch.distributed.new_group(ranks)
    #         if rank in ranks:
    #             _DATA_PARALLEL_GROUP = group
    #             _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    if _MODEL_PARALLEL_GROUP is not None:
        logger.error("model parallel group is already initialized")
    for i in range(data_parallel_size):
        ranks = [
            data_parallel_group_ranks[i]
            for data_parallel_group_ranks in all_data_parallel_group_ranks
        ]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    if _TENSOR_MODEL_PARALLEL_GROUP is not None:
        logger.error("tensor model parallel group is already initialized")
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(
            i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
        )
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    if _PIPELINE_MODEL_PARALLEL_GROUP is not None:
        logger.error("pipeline model parallel group is already initialized")

    for ranks in pipeline_parallel_topo:
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            # group = torch.distributed.new_group(ranks)
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    # for i in range(num_pipeline_model_parallel_groups):
    #     ranks = range(i, world_size, num_pipeline_model_parallel_groups)
    #     group = torch.distributed.new_group(ranks)
    #     if rank in ranks:
    #         _PIPELINE_MODEL_PARALLEL_GROUP = group
    #         _PIPELINE_GLOBAL_RANKS = ranks

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    if _MODEL_PARALLEL_GROUP is None:
        logger.error("model parallel group is not initialized")
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        logger.error("intra_layer_model parallel group is not initialized")
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    if _PIPELINE_MODEL_PARALLEL_GROUP is None:
        logger.error("pipeline_model parallel group is not initialized")
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    if _DATA_PARALLEL_GROUP is None:
        logger.error("data parallel group is not initialized")
    return _DATA_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if (
            virtual_pipeline_model_parallel_world_size is not None
            and get_virtual_pipeline_model_parallel_rank()
            != (virtual_pipeline_model_parallel_world_size - 1)
        ):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next stage executes encoder block for a model with both encoder and decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(
        rank + 1
    )


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    if _DATA_PARALLEL_GLOBAL_RANKS is None:
        logger.error("Data parallel group is not initialized")
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the current tensor parallel group."""
    if _PIPELINE_GLOBAL_RANKS is None:
        logger.error("Pipeline parallel group is not initialized")
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the current tensor parallel group."""
    if _PIPELINE_GLOBAL_RANKS is None:
        logger.error("Pipeline parallel group is not initialized")
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline."""
    if _PIPELINE_GLOBAL_RANKS is None:
        logger.error("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline."""
    if _PIPELINE_GLOBAL_RANKS is None:
        logger.error("Pipeline parallel group is not initialized")
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    if _GLOBAL_MEMORY_BUFFER is not None:
        logger.error("global memory buffer is already initialized")
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object."""
    if _GLOBAL_MEMORY_BUFFER is None:
        logger.error("global memory buffer is not initialized")
    return _GLOBAL_MEMORY_BUFFER


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

# -*- coding: utf-8 -*-
"""The utility of GPU Affinity."""

import collections
import itertools
import logging
import os
import pathlib
import re
from enum import Enum
from enum import auto
from typing import Any
from typing import Union

import pynvml


logger = logging.getLogger(__name__)


class AffinityMode(Enum):
    """Affinity Mode."""

    NONE = auto()
    SOCKET = auto()
    SOCKET_AUTO = auto()
    SOCKET_SINGLE = auto()
    SOCKET_SINGLE_UNIQUE = auto()
    SOCKET_UNIQUE_INTERLEAVED = auto()
    SOCKET_UNIQUE_CONTIGUOUS = auto()


class Device:
    """Manage GPU Device."""

    # assume nvml returns list of 64 bit ints
    _nvml_bit_affinity = 64
    cpu_count: Any = 0 if os.cpu_count() is None else os.cpu_count()
    _nvml_affinity_elements = (cpu_count + _nvml_bit_affinity - 1) // _nvml_bit_affinity

    def __init__(self, device_idx):
        """Init."""
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self):
        """Return Device Name."""
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self):
        """Return Device UUID."""
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self):
        """Return CPU info related to GPU."""
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(
            self.handle, Device._nvml_affinity_elements
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = f"{j:064}" + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


def get_thread_siblings_list():
    """Return representing pairs of hyperthreading cores.

    Returns a list of 2-element integer tuples representing pairs \
      of hyperthreading cores.
    """
    path = "/sys/devices/system/cpu/cpu*/topology/thread_siblings_list"
    thread_siblings_list = []
    pattern = re.compile(r"(\d+)\D(\d+)")
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname, encoding="utf-8") as fin:
            content = fin.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(sorted(map(int, res[0])))
                thread_siblings_list.append(pair)
    thread_siblings_list = list(set(thread_siblings_list))
    return thread_siblings_list


def build_thread_siblings_dict(siblings_list):
    """Build thread_siblings_dict."""
    siblings_dict = {}
    for siblings_tuple in siblings_list:
        for core in siblings_tuple:
            siblings_dict[core] = siblings_tuple

    return siblings_dict


def group_list_by_dict(affinity, siblings_dict):
    """Group list by dict."""
    sorted_affinity = sorted(affinity, key=lambda x: siblings_dict.get(x, (x,)))
    grouped = itertools.groupby(
        sorted_affinity, key=lambda x: siblings_dict.get(x, (x,))
    )
    grouped_affinity = []
    for _, group in grouped:
        grouped_affinity.append(tuple(group))
    return grouped_affinity


def group_affinity_by_siblings(socket_affinities):
    """Group affinity by siblings."""
    siblings_list = get_thread_siblings_list()
    siblings_dict = build_thread_siblings_dict(siblings_list)

    grouped_socket_affinities = []

    for socket_affinity in socket_affinities:
        grouped_socket_affinities.append(
            group_list_by_dict(socket_affinity, siblings_dict)
        )
    return grouped_socket_affinities


def ungroup_affinities(affinities, cores):
    """Ungroup affinities."""
    ungrouped_affinities = []

    for affinity in affinities:
        if cores == "all_logical":
            ungrouped_affinities.append(list(itertools.chain(*affinity)))
        elif cores == "single_logical":
            ungrouped_affinities.append([group[0] for group in affinity])
        else:
            raise RuntimeError("Unknown cores mode")
    return ungrouped_affinities


def check_socket_affinities(socket_affinities):
    """Check socket affinities."""
    # sets of cores should be either identical or disjoint

    for i, j in itertools.product(socket_affinities, socket_affinities):
        if not set(i) == set(j) and not set(i).isdisjoint(set(j)):
            raise RuntimeError(
                f"Sets of cores should be either identical or disjoint, "
                f"but got {i} and {j}."
            )
            # logger.warning(
            #     "Sets of cores should be either identical or disjoint, "
            #     "but got %s and %s.",
            #     i,
            #     j,
            # )


def get_socket_affinities(nproc_per_node, exclude_unavailable_cores=True):
    """Get socket affinities."""
    devices = [Device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.get_cpu_affinity() for dev in devices]
    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        socket_affinities = [
            list(set(affinity) & available_cores) for affinity in socket_affinities
        ]

    check_socket_affinities(socket_affinities)

    return socket_affinities


def get_grouped_socket_affinities(nproc_per_node, exclude_unavailable_cores=True):
    """Get grouped socket affinities."""
    socket_affinities = get_socket_affinities(nproc_per_node, exclude_unavailable_cores)
    grouped_socket_affinities = group_affinity_by_siblings(socket_affinities)
    return grouped_socket_affinities


def set_socket_affinity_auto(gpu_id, nproc_per_node):
    """Set socket affinity autoly."""
    available_cores = os.sched_getaffinity(0)
    available_cores_num = len(available_cores)
    group_cores_num = available_cores_num // nproc_per_node
    ungrouped_affinities = [
        [x + i * group_cores_num for x in range(group_cores_num)]
        for i in range(nproc_per_node)
    ]
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])


def set_socket_affinity(gpu_id, nproc_per_node, cores):
    """Set socket affinity.

    The process is assigned with all available physical CPU cores from the CPU
    socket connected to the GPU with a given id.

    Args:
        gpu_id: index of a GPU
        nproc_per_node: number of processes per node
        cores: 'all_logical' or 'single_logical'
    """
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    ungrouped_affinities = ungroup_affinities(grouped_socket_affinities, cores)
    # ungrouped_affinities = [[x + i * 16 for x in range(16)] for i in range(8)]
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])


def set_socket_single_affinity(gpu_id, nproc_per_node, cores):
    """Set socket single affinity.

    The process is assigned with the first available physical CPU core from the
    list of all CPU physical cores from the CPU socket connected to the GPU with
    a given id.

    Args:
        gpu_id: index of a GPU
        nproc_per_node: number of processes per node
        cores: 'all_logical' or 'single_logical'
    """
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)
    single_grouped_socket_affinities = [
        group[:1] for group in grouped_socket_affinities
    ]
    ungrouped_affinities = ungroup_affinities(single_grouped_socket_affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])


def set_socket_single_unique_affinity(gpu_id, nproc_per_node, cores):
    """Set socket single unique affinity.

    The process is assigned with a single unique available physical CPU core
    from the list of all CPU cores from the CPU socket connected to the GPU with
    a given id.

    Args:
        gpu_id: index of a GPU
        nproc_per_node: number of processes per node
        cores: 'all_logical' or 'single_logical'
    """
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)

    affinities = []
    assigned_groups = set()

    for grouped_socket_affinity in grouped_socket_affinities:
        for group in grouped_socket_affinity:
            if group not in assigned_groups:
                affinities.append([group])
                assigned_groups.add(group)
                break

    ungrouped_affinities = ungroup_affinities(affinities, cores)

    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])


def set_socket_unique_affinity(gpu_id, nproc_per_node, cores, mode, balanced=True):
    """Set socket unique affinity.

    The process is assigned with a unique subset of available physical CPU
    cores from the CPU socket connected to a GPU with a given id.
    Assignment automatically includes hyperthreading siblings (if siblings are
    available).

    Args:
        gpu_id: index of a GPU
        nproc_per_node: number of processes per node
        cores: 'all_logical' or 'single_logical'
        mode: 'contiguous' or 'interleaved'
        balanced: assign an equal number of physical cores to each process,
    """
    grouped_socket_affinities = get_grouped_socket_affinities(nproc_per_node)

    grouped_socket_affinities_to_device_ids = collections.defaultdict(list)

    for idx, grouped_socket_affinity in enumerate(grouped_socket_affinities):
        grouped_socket_affinities_to_device_ids[tuple(grouped_socket_affinity)].append(
            idx
        )

    # compute minimal number of physical cores per GPU across all GPUs and
    # sockets, code assigns this number of cores per GPU if balanced == True
    cores_per_gpu = []
    for cores_info, gpus in grouped_socket_affinities_to_device_ids.items():
        cores_per_gpu.append(len(cores_info) // len(gpus))
    min_physical_cores_per_gpu = min(cores_per_gpu)

    grouped_unique_affinities = [None] * nproc_per_node

    for (
        grouped_socket_affinity,
        device_ids,
    ) in grouped_socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            grouped_socket_affinity = grouped_socket_affinity[
                : devices_per_group * min_physical_cores_per_gpu
            ]
        else:
            cores_per_device = len(grouped_socket_affinity) // devices_per_group

        for socket_subgroup_id, device_id in enumerate(device_ids):
            # In theory there should be no difference in performance between
            # 'interleaved' and 'contiguous' pattern on Intel-based DGX-1,
            # but 'contiguous' should be better for DGX A100 because on AMD
            # Rome 4 consecutive cores are sharing L3 cache.
            # Attention: code doesn't attempt to automatically detect layout of
            # L3 cache, also external environment may already exclude some
            # cores, this code makes no attempt to detect it and to align
            # mapping to multiples of 4.

            if mode == "interleaved":
                unique_grouped_affinity = list(
                    grouped_socket_affinity[socket_subgroup_id::devices_per_group]
                )
            elif mode == "contiguous":
                unique_grouped_affinity = list(
                    grouped_socket_affinity[
                        socket_subgroup_id
                        * cores_per_device : (socket_subgroup_id + 1)
                        * cores_per_device
                    ]
                )
            else:
                raise RuntimeError("Unknown set_socket_unique_affinity mode")

            grouped_unique_affinities[device_id] = unique_grouped_affinity

    ungrouped_affinities = ungroup_affinities(grouped_unique_affinities, cores)
    os.sched_setaffinity(0, ungrouped_affinities[gpu_id])


def set_affinity(
    gpu_id,
    nproc_per_node=None,
    *,
    mode: Union[str, AffinityMode] = AffinityMode.SOCKET_AUTO,
    cores="all_logical",
    balanced=True,
):
    """Set affinity.

    The process is assigned with a proper CPU affinity that matches CPU-GPU
    hardware architecture on a given platform. Usually, it improves and
    stabilizes the performance of deep learning training workloads.
    This function assumes that the workload runs in multi-process single-device
    mode (there are multiple training processes, and each process is running on
    a single GPU). This is typical for multi-GPU data-parallel training
    workloads (e.g., using `torch.nn.parallel.DistributedDataParallel`).
    Available affinity modes:
    * 'socket' - the process is assigned with all available physical CPU cores
    from the CPU socket connected to the GPU with a given id.
    * 'socket_single' - the process is assigned with the first available
    physical CPU core from the list of all CPU cores from the CPU socket
    connected to the GPU with a given id (multiple GPUs could be assigned with
    the same CPU core).
    * 'socket_single_unique' - the process is assigned with a single unique
    available physical CPU core from the list of all CPU cores from the CPU
    socket connected to the GPU with a given id.
    * 'socket_unique_interleaved' - the process is assigned with a unique
    subset of available physical CPU cores from the CPU socket connected to a
    GPU with a given id, cores are assigned with interleaved indexing pattern
    * 'socket_unique_contiguous' - (the default) the process is assigned with a
    unique subset of available physical CPU cores from the CPU socket connected
    to a GPU with a given id, cores are assigned with contiguous indexing
    pattern
    Available "cores" modes:
    * 'all_logical' - assigns the process with all logical cores associated with
    a given corresponding physical core (i.e., automatically includes all
    available hyperthreading siblings)
    * 'single_logical' - assigns the process with only one logical core
    associated with a given corresponding physical core (i.e., excludes
    hyperthreading siblings)
    'socket_unique_contiguous' is the recommended mode for deep learning
    training workloads on NVIDIA DGX machines.

    Args:
        gpu_id: integer index of a GPU, value from 0 to 'nproc_per_node' - 1
        nproc_per_node: number of processes per node
        mode: affinity mode
        balanced: assign an equal number of physical cores to each process,
            affects only 'socket_unique_interleaved' and
            'socket_unique_contiguous' affinity modes
        cores: 'all_logical' or 'single_logical'
    Returns a set of logical CPU cores on which the process is eligible to run.

    Example:
    import argparse
    import os
    import gpu_affinity
    import torch
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--local_rank',
            type=int,
            default=os.getenv('LOCAL_RANK', 0),
        )
        args = parser.parse_args()
        nproc_per_node = torch.cuda.device_count()
        affinity = gpu_affinity.set_affinity(args.local_rank, nproc_per_node)
        print(f'{args.local_rank}: core affinity: {affinity}')
    if __name__ == "__main__":
        main()
    Launch the example with:
    python -m torch.distributed.launch --nproc_per_node <#GPUs> example.py
    WARNING: On DGX A100, only half of the CPU cores have direct access to GPUs.
    This function restricts execution only to the CPU cores directly connected
    to GPUs, so on DGX A100, it will limit the code to half of the CPU cores and
    half of CPU memory bandwidth (which may be fine for many DL models).
    WARNING: Intel's OpenMP implementation resets affinity on the first call to
    an OpenMP function after a fork. It's recommended to run with env variable:
    `KMP_AFFINITY=disabled` if the affinity set by gpu_affinity should be
    preserved after a fork (e.g. in PyTorch DataLoader workers).
    """
    if not isinstance(mode, AffinityMode):
        mode = AffinityMode[mode]
    pynvml.nvmlInit()
    if nproc_per_node is None:
        nproc_per_node = pynvml.nvmlDeviceGetCount()

    if mode == AffinityMode.NONE:
        pass
    elif mode == AffinityMode.SOCKET:
        set_socket_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.SOCKET_SINGLE:
        set_socket_single_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.SOCKET_SINGLE_UNIQUE:
        set_socket_single_unique_affinity(gpu_id, nproc_per_node, cores)
    elif mode == AffinityMode.SOCKET_UNIQUE_INTERLEAVED:
        set_socket_unique_affinity(
            gpu_id, nproc_per_node, cores, "interleaved", balanced
        )
    elif mode == AffinityMode.SOCKET_UNIQUE_CONTIGUOUS:
        set_socket_unique_affinity(
            gpu_id, nproc_per_node, cores, "contiguous", balanced
        )
    elif mode == AffinityMode.SOCKET_AUTO:
        set_socket_affinity_auto(gpu_id, nproc_per_node)
    else:
        raise RuntimeError("Unknown affinity mode")

    affinity = os.sched_getaffinity(0)
    return affinity

# -*- coding: utf-8 -*-
"""Summary: dist_util.py.

dist_util.py provides distributed atomic operation and query of distributed system
information.
"""
import logging
import os
import socket
import time
from typing import Any
from typing import Optional

import torch

from unirobot.utils.constants import LaunchMode
from unirobot.utils.settings import global_settings
from unirobot.utils.settings import settings


GPUS_PER_NODE = 8
_MPI_AVAILABLE = True

try:
    from mpi4py import MPI
except ImportError:
    _MPI_AVAILABLE = False


logger = logging.getLogger(__name__)


def build_machine_connection(
    master_host_name: str,
    port: int,
    node_rank: int,
) -> int:
    """
    Establish communication between nodes to ensure stable information transmission.

    Args:
      master_host_name (str): master_host_name is master's host name.\
                              All slave nodes establish connections based on it.
      port (int): port is used for communication between nodes.
      node_rank (int): node_rank is index of machine node.

    Returns:
      (int): If successful, the return operation status is 0. If unsuccessful, the loop
      is executed.

    Example:
    literal blocks::

      from unirobot.utils.dist_util import build_machine_connection

      status = build_machine_connection(master_host_name, port, node_rank)
    """
    if node_rank == 0:
        # make sure the hostname is setup properly.
        while True:
            try:
                host = socket.gethostbyname(master_host_name)
                if host is not None:
                    return 0
            except OSError:
                logger.exception("Host `%s` is not ready. Try in 5 seconds.", host)
                time.sleep(5)
    else:
        # make sure the root rank has set up the communication channel properly.
        while True:
            a_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result_of_check = a_socket.connect_ex((master_host_name, port))
                if result_of_check == 0:
                    return 0
                time.sleep(5)
                logger.info("Waiting root rank to be ready.")
            except OSError:
                logger.error(
                    "%s:%s is not ready. Try in 5 seconds.",
                    master_host_name,
                    port,
                )
                time.sleep(5)


def is_mpi_available() -> bool:
    """Check whether openmpi is available."""
    return _MPI_AVAILABLE


def get_master_hostname() -> str:
    """Get master host name."""
    return global_settings.vc_master_hosts


def get_global_rank() -> int:
    """Get global rank for different launch mode."""
    if settings.launch_mode == LaunchMode.MPI:
        if is_mpi_available():
            comm = MPI.COMM_WORLD
            return comm.Get_rank()
        return 0
    # spawn need a fake value to init filemanager
    return 0


def get_local_rank() -> int:
    """Get local rank for spawn launch mode."""
    local_rank = get_global_rank() % torch.cuda.device_count()
    return local_rank


def get_world_size() -> int:
    """Get world size for different launch mode."""
    if settings.launch_mode == LaunchMode.MPI:
        if is_mpi_available():
            comm = MPI.COMM_WORLD
            return comm.Get_size()
        return 1
    raise NotImplementedError()


def dist_barrier() -> Any:
    """Distributed barrier."""
    if is_mpi_available():
        comm = MPI.COMM_WORLD
        comm.Barrier()


def dist_abort() -> Any:
    """Distributed abort."""
    if is_mpi_available():
        comm = MPI.COMM_WORLD
        comm.Abort(1)


def broadcast_obj(obj: Any) -> Any:
    """Broadcast obj from master to workers."""
    if is_mpi_available():
        comm = MPI.COMM_WORLD
        return comm.bcast(obj)
    return obj


def init_set_device():
    """Set device manually in torchpilot init. Default launch mode is MPI."""
    if is_mpi_available():
        comm = MPI.COMM_WORLD
        if comm.Get_size() <= 1 or global_settings.cuda_visible_devices:
            return
        global_rank = comm.Get_rank()
        local_rank = global_rank % GPUS_PER_NODE
        torch.cuda.set_device(local_rank)


def disable_kmp_affinity():
    """Set KMP_AFFINITY as Disable."""
    os.environ["KMP_AFFINITY"] = "disabled"


class DistSpawnInfo:
    """Distributed infomation for spawn launch mode."""

    def __init__(self, world_size: int) -> None:
        """Initialize."""
        if world_size > 8 and world_size % 8 != 0:
            raise ValueError(
                "When run on multi node, world size needs to be multiple of 8, "
                f"but gets {world_size}"
            )
        self._word_size = world_size
        self._local_rank: Optional[int] = None

    def set_local_rank(self, local_rank: int) -> Any:
        """Set local rank."""
        self._local_rank = local_rank

    @staticmethod
    def get_node_rank() -> int:
        """Get node rank, the index of machine node."""
        host_name = os.environ.get("HOSTNAME", "")
        node_attr = host_name.split("-")[-2]
        node_index = host_name.split("-")[-1]
        if node_attr == "master":
            node_rank = int(node_index)
        else:
            node_rank = int(node_index) + 1
        return node_rank

    @staticmethod
    def get_nnodes() -> int:
        """Get nnodes, the number of machine."""
        master_num = int(os.environ.get("VC_MASTER_NUM", "0"))
        worker_num = int(os.environ.get("VC_WORKER_NUM", 0))
        nnodes = master_num + worker_num
        return nnodes

    def get_global_rank(self) -> int:
        """Get global rank."""
        node_rank = self.get_node_rank()
        nproc = self.get_gpus_per_node()
        if isinstance(self._local_rank, int):
            global_rank = node_rank * nproc + self._local_rank
            return global_rank
        raise RuntimeError("Not set local rank yet.")

    def get_world_size(self) -> int:
        """Get world size."""
        return self._word_size

    def get_gpus_per_node(self) -> int:
        """Get gpus per node."""
        if self._word_size <= 8:
            return self._word_size
        return self._word_size // self.get_nnodes()

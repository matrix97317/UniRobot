# -*- coding: utf-8 -*-
"""Run for Brain DDP Launcher."""

import datetime
import logging
import random
import sys
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import torch

from unirobot.utils.cfg_parser import PyConfig
from unirobot.utils.constants import Backend
from unirobot.utils.constants import LaunchMode
from unirobot.utils.dist_util import DistSpawnInfo
from unirobot.utils.dist_util import broadcast_obj
from unirobot.utils.dist_util import build_machine_connection
from unirobot.utils.dist_util import get_global_rank
from unirobot.utils.dist_util import get_master_hostname
from unirobot.utils.dist_util import get_world_size
from unirobot.utils.dist_util import is_mpi_available
from unirobot.utils.exceptions import exception_hook
from unirobot.utils.file_util import FileUtil
from unirobot.utils.gpu_affinity import set_affinity
from unirobot.utils.log_util import log_file_init
from unirobot.utils.slot_loader import load_brain_slot
from unirobot.utils.settings import settings
from unirobot.utils.system_info import show_hardware_info
from unirobot.utils.unirobot_slot import INFERRER
from unirobot.utils.unirobot_slot import TRAINER


logger = logging.getLogger(__name__)

DATETIME_FORMAT = "%Y%m%dT%H%M%S"


def mpi_run(
    init_method: str,
    rank: int,
    world_size: int,
    runtime_func: Callable[[Dict[str, Any]], None],
    runtime_func_kwargs: Dict[str, Any],
) -> None:
    """
    Run a function from a child process.

    Args:
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        rank (int): the index of all GPU process.
        world_size (int): the number of all GPU process.
        runtime_func: functions for distributed parallel execution.It should conform
            to the STMD (Single Threading Multi Data) paradigm.
        runtime_func_kwargs (Dict[str, Any]): Params for runtime funcion.
    """
    try:
        torch.distributed.init_process_group(
            backend=settings.backend.value,
            timeout=datetime.timedelta(seconds=settings.com_timeout),
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.warning("Set communication timeout %d s.", settings.com_timeout)
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception("Failed to init process group with mpi.")
        raise ex

    if get_world_size() > 1:
        local_rank = rank % torch.cuda.device_count()
        affinity = set_affinity(local_rank)
        logger.warning("GPU process %d affinity: %s", rank, affinity)
    runtime_func(**runtime_func_kwargs)  # type: ignore[call-arg]
    torch.distributed.barrier()
    logger.warning("Destroying process group...")
    torch.distributed.destroy_process_group()
    logger.warning("The program has finished running. You can use `Ctrl+C` to close it")


def spawn_run(
    local_rank: int,
    world_size: int,
    init_method: str,
    runtime_func: Callable[[Dict[str, Any]], None],
    runtime_func_kwargs: Dict[str, Any],
    backend: str,
) -> None:
    """Run a function from a child process.

    Args:
        local_rank (int): rank of the current process on the current machine.
        world_size (int): number of all processes.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        runtime_func: functions for distributed parallel execution.It should conform
            to the STMD ( Single Threading Multi Data) paradigm.
        runtime_func_kwargs (Dict[str, Any]): Params for runtime funcion.
        backend (str): backend used in torch.distributed.init_process_group.
    """
    dist_spawn_info = DistSpawnInfo(world_size)
    dist_spawn_info.set_local_rank(local_rank)
    global_rank = dist_spawn_info.get_global_rank()

    # Initialize filemaneger, log and reg registrie for current process
    cfg = runtime_func_kwargs["cfg"]
    run_name = runtime_func_kwargs["run_name"]
    resume = runtime_func_kwargs["resume"]
    FileUtil.init(cfg, run_name, resume, global_rank)
    log_file_init("unirobot", global_rank)
    load_brain_slot()

    # Initialize the process group.
    try:
        torch.distributed.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=settings.com_timeout),
            init_method=init_method,
            world_size=world_size,
            rank=global_rank,
        )
        logger.warning("Set communication timeout %d s.", settings.com_timeout)
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception("Failed to init process group with spawn.")
        raise ex
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        affinity = set_affinity(local_rank)
        logger.warning("GPU process %d affinity: %s", global_rank, affinity)
    runtime_func(**runtime_func_kwargs)  # type: ignore[call-arg]
    torch.distributed.barrier()
    logger.warning("Destroying process group...")
    torch.distributed.destroy_process_group()
    logger.warning("The program has finished running. You can use `Ctrl+C` to close it")


def mpi_launcher(
    init_method: str,
    mpi_rank: int,
    mpi_world_size: int,
    runtime_func: Callable[..., None],
    runtime_func_kwargs: Dict[str, Any],
) -> None:
    """
    Start the STMD paradigm program based on MPI mode.

    Args:
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        mpi_rank (int): the index of all MPI process.
        mpi_world_size (int): the number of all MPI process.
        runtime_func: functions for distributed parallel execution.It should conform
            to the STMD (Single Threading Multi Data) paradigm.
        runtime_func_kwargs (Dict[str, Any]): Params for runtime funcion.
    """
    mpi_run(
        init_method,
        mpi_rank,
        mpi_world_size,
        runtime_func,
        runtime_func_kwargs,
    )


def spawn_launcher(
    world_size: int,
    init_method: str,
    runtime_func: Callable[..., None],
    runtime_func_kwargs: Dict[str, Any],
    backend: str,
) -> None:
    """Start the STMD paradigm program based on torch.multiprocessing.spawn mode.

    Args:
        world_size (int): number of all processes.
        init_method (string): method to initialize the distributed training.
            TCP initialization: requiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        runtime_func (callable): functions for distributed parallel execution.It should
            conform to the STMD (Single Threading Multi Data) paradigm.
        runtime_func_kwargs (Dict[str, Any]): Params for runtime funcion.
        backend (str): backend used in torch.distributed.init_process_group.
    """
    dist_spawn_info = DistSpawnInfo(world_size)
    nproc = dist_spawn_info.get_gpus_per_node()
    # each gpu spawn one process
    torch.multiprocessing.spawn(
        spawn_run,
        nprocs=nproc,
        args=(
            world_size,
            init_method,
            runtime_func,
            runtime_func_kwargs,
            backend,
        ),
        daemon=False,
    )


def runtime_program(
    cfg: str,
    run_name: str,
    resume: bool,
    infer_type: str,
    export_type: str,
    ckpt: Union[None, str, List[str]],
    dataset_mode: str,
) -> None:
    """Add program that can be runned in distribution way.

    Args:
        cfg (str): The path to config file.
        run_name (str): Exp run name.
        resume (bool): Whether to resume.
        infer_type (str): The type of infer model file, support [torch, trace, tvm].
        export_type (str): The type of export model file, support [trace].
        ckpt (str): The ckpt list for inferring.
        dataset_mode (str): Which mode specified to dataset.
    """
    # print("----- Demo Code ------")
    # parser cfg
    config_cfg = PyConfig.fromfile(cfg)
    args = {
        "cfg": config_cfg,
        "run_name": run_name,
        "resume": resume,
    }

    if infer_type is not None or export_type is not None:
        args.update(config_cfg.infer)
        args["infer_type"] = infer_type
        args["export_type"] = export_type
        if dataset_mode:
            logger.warning(
                "Specify '%s' as dataset mode.",
                dataset_mode,
            )
        # config_cfg.dataloader["dataset_cfg"]["mode"] = 'train'
        if ckpt:
            args["eval_ckpt_list"] = list(ckpt)
        inferrer = INFERRER.build(args)
        inferrer.infer()
    else:
        args.update(config_cfg.trainer)
        if dataset_mode:
            logger.warning(
                "Specify '%s' as dataset mode.",
                dataset_mode,
            )
            config_cfg.dataloader["dataset_cfg"]["mode"] = dataset_mode
        trainer = TRAINER.build(args)
        trainer.train()


def run(
    device_num: int,
    config: str,
    run_name: str,
    infer_type: str,
    export_type: str,
    ckpt: Union[None, str, List[str]],
    resume: bool,
    port: int,
    dataset_mode: str,
) -> None:
    """Entry for unirobot."""
    # Set except hook when run ddp_launcher.
    sys.excepthook = exception_hook

    logger.debug("Args before broadcast, run_name: %s, port: %s", run_name, port)
    if not run_name:
        run_name = broadcast_obj(time.strftime(DATETIME_FORMAT, time.gmtime()))
    if not port:
        port = broadcast_obj(random.SystemRandom().randint(1024, 65535))
    logger.debug("Args after broadcast, run_name: %s, port: %s", run_name, port)

    # NOTE: MPI mode gets real global rank, while spawn mode gets fake
    # global_rank=0, just for initialize FileManager and logger.
    global_rank = get_global_rank()
    FileUtil.init(config, run_name, resume, global_rank)
    log_file_init("unirobot", global_rank)
    load_brain_slot()
    show_hardware_info()

    logger.warning("Running with launch mode: `%s`.", settings.launch_mode)

    if settings.backend is None:
        if torch.distributed.is_nccl_available():
            settings.backend = Backend.NCCL
        elif torch.distributed.is_mpi_available() and is_mpi_available():
            settings.backend = Backend.MPI
        else:
            settings.backend = Backend.GLOO
    logger.warning("Running with distributed backend: `%s`.", settings.backend)

    args = {
        "cfg": config,
        "run_name": run_name,
        "resume": resume,
        "infer_type": infer_type,
        "export_type": export_type,
        "ckpt": ckpt,
        "dataset_mode": dataset_mode,
    }
    # launch distribution program.
    if settings.launch_mode == LaunchMode.SPAWN:
        master_hostname = get_master_hostname()
        node_rank = DistSpawnInfo.get_node_rank()
        status = build_machine_connection(
            master_hostname,
            port,
            node_rank,
        )
        if status == 0:
            init_method = f"tcp://{master_hostname}:{port}"
            spawn_launcher(
                device_num,
                init_method,
                runtime_program,
                args,
                settings.backend.value,
            )
    elif settings.launch_mode == LaunchMode.MPI:
        master_hostname = get_master_hostname()
        global_rank = get_global_rank()
        world_size = get_world_size()
        init_method = f"tcp://{master_hostname}:{port}"
        mpi_launcher(
            init_method,
            global_rank,
            world_size,
            runtime_program,
            args,
        )

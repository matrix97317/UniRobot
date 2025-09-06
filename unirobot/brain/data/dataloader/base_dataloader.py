# -*- coding: utf-8 -*-
"""UniRobot DataLoader."""

import logging
import math
import os
import random
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader

from unirobot.brain.data.dataset.base_dataset import BaseDataset
from unirobot.brain.data.sampler.base_distributed_sampler import URDistributedSampler
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_rank,
)
from unirobot.utils.unirobot_slot import DATASET
from unirobot.utils.unirobot_slot import SAMPLER


logger = logging.getLogger(__name__)


class URDataLoader(DataLoader):
    """UniRobot DataLoader.

    Combines a dataset and a sampler, and provides an iterable over the given dataset.

    The `DataLoader` supports both map-style and iterable-style datasets with
    single- or multi-process loading, customizing loading order and optional automatic
    batching (collation) and memory pinning.

    Args:
        dataset_cfg (Dataset): dataset from which to load the data.
        sampler_cfg (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a
            custom type, see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch size,
            then the last batch will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a
            batch from workers. Should always be non-negative. (default: ``0``)
        prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers samples prefetched across all workers. (default: ``2``)
        persistent_workers (bool, optional): If ``True``, the data loader will not
            shutdown the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            (default: ``False``)
        seed (int, Optional): Seed to be used. Default: None.
        to_cuda (bool, Optional): Whether send data to cuda. Default: False.
        recursive_to_cuda (bool, Optional): Whether send data to cuda recursively.
            (default: ``False``)

    .. warning::
        If the ``spawn`` start method is used, :attr:`worker_init_fn`
        cannot be an unpicklable object, e.g., a lambda function. See
        :ref:``multiprocessing-best-practices`` on more details related
        to multiprocessing in PyTorch.

    .. warning::
        ``len(dataloader)`` heuristic is based on the length of the sampler used.
        When :attr:``dataset`` is an :class:``~torch.utils.data.IterableDataset``,
        it instead returns an estimate based on ``len(dataset) / batch_size``, with
        proper rounding depending on :attr:`drop_last`, regardless of multi-process
        loading configurations. This represents the best guess PyTorch can make because
        PyTorch trusts user :attr:`dataset` code in correctly handling multi-process
        loading to avoid duplicate data.

        However, if sharding results in multiple workers having incomplete last batches,
        this estimate can still be inaccurate, because (1) an otherwise complete batch
        can be broken into multiple ones and (2) more than one batch worth of samples
        can be dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not
        detect such cases in general.

        See ``Dataset Types_`` for more details on these two types of datasets and how
        :class:~torch.utils.data.IterableDataset interacts with Multi-process data
        loading.

    .. warning::
        See :ref:``reproducibility``, and :ref:``dataloader-workers-random-seed``, and
        :ref:``data-loading-randomness`` notes for random seed related questions.

    """

    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        sampler_cfg: Dict[str, Any],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        multiprocessing_context=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        seed: Optional[int] = None,
        to_cuda: Optional[bool] = False,
        use_channel_last: bool = False,
        recursive_to_cuda: Optional[bool] = False,
    ):
        """Init."""
        dataset = DATASET.build(dataset_cfg)
        if not isinstance(dataset, BaseDataset):
            raise ValueError(
                f"Expect type(dataset)=BaseDataset but got {type(dataset)}."
            )
        collate_fn = dataset.batch_collate
        if not callable(collate_fn):
            raise ValueError("Expect callable(collate_fn)=True but got False.")

        self._use_channel_last = use_channel_last

        # Fake batch size.
        task_batch_size = batch_size
        if isinstance(batch_size, dict):
            batch_size = 1

        sampler_cfg["dataset"] = dataset
        sampler_cfg["batch_size"] = task_batch_size
        sampler = SAMPLER.build(sampler_cfg)
        if not isinstance(sampler, URDistributedSampler):
            raise ValueError(
                f"Expect type(sampler)=URDistributedSampler but got {type(sampler)}."
            )

        if sampler.drop_last != drop_last:
            raise ValueError("Inconsistent `drop_last` in sampler and dataloader.")

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=_init_worker_funcion(workers_per_gpu=1, seed=seed),
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.actual_data_size = sampler.actual_data_size
        self._data_iter = iter(self)
        self._step = 0
        self._to_cuda = to_cuda
        self._recursive_to_cuda = recursive_to_cuda
        self._pin_memory = pin_memory

    def get_sample_per_epoch(self) -> int:
        """Get sample per epoch."""
        indices = self.sampler.get_indices()  # type: ignore[attr-defined]
        return len(indices) if indices else self.actual_data_size

    def get_sample_num_per_gpu(self) -> int:
        """Get num samples."""
        return len(self.sampler)  # type: ignore[arg-type]

    def get_batch_size_per_gpu(self) -> Optional[int]:
        """Get batch size per gpu."""
        return self.batch_size

    def get_one_epoch_step_per_gpu(self) -> int:
        """Get one epoch step per gpu."""
        if self.batch_size is None:
            raise ValueError("Expect self.batch_size is not None, but got None.")
        data_per_gpu = len(self.sampler)  # type: ignore[arg-type]
        if not self.drop_last and data_per_gpu % self.batch_size != 0:
            return math.ceil(data_per_gpu / self.batch_size)
        return int(data_per_gpu / self.batch_size)

    def get_state(self) -> Tuple[int, int, List[int]]:
        """Get dataloader state. i.e. `epoch`, `step`, `sampler.indices`."""
        return (
            self.sampler.get_epoch(),  # type: ignore[attr-defined]
            self._step,
            self.sampler.get_indices(),  # type: ignore[attr-defined]
        )

    def set_state(
        self,
        epoch: int,
        step: int = 0,
        indices: Optional[List[int]] = None,
    ):
        """Set state.

        Args:
            epoch (int): The epoch.
            step (int): The step.
                Default: `0`.
            indices (List[int], optional): If set, will use this instead of the default
                `list(range(len(self.dataset)))` of sampler.
                Default: `None`.
        """
        if not isinstance(self.sampler, URDistributedSampler):
            raise ValueError(
                f"Expect type(sampler)=`URDistributedSampler` but got "
                f"type(sampler)={type(self.sampler)}."
            )

        # Epoch.
        self.sampler.set_epoch(epoch)

        # Step.
        self.sampler.set_step(step)

        # Indices.
        if indices is not None:
            self.sampler.set_indices(indices)

        # Reset.
        self.reset(step)

    def reset(self, step: int = 0) -> None:
        """Reset data iter."""
        self._data_iter = iter(self)
        self._step = step

    def recursive_to_cuda(self, data_to_recursive):
        """Move data to cuda recursively.

        Args:
            data_to_recursive: Data to move to cuda recursively.

        Returns:
            data_to_recursive: Data on cuda.
        """
        if isinstance(data_to_recursive, torch.Tensor):
            if self._pin_memory:
                data_to_recursive = data_to_recursive.pin_memory()
            data_to_recursive = data_to_recursive.cuda(non_blocking=True)
            return data_to_recursive
        if isinstance(data_to_recursive, dict):
            for key, data_item in data_to_recursive.items():
                data_to_recursive[key] = self.recursive_to_cuda(data_item)
        elif isinstance(data_to_recursive, list):
            for i, data_item in enumerate(data_to_recursive):
                data_to_recursive[i] = self.recursive_to_cuda(data_item)
        else:
            pass
            # logger.warning("[recursive_to_cuda]
            #  Dont't support type: %s",type(data_to_recursive))
        return data_to_recursive

    def move_data_to_cuda(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move data to cuda.

        Args:
            data Dict[str, Any]: Data to move.

        Returns:
            data Dict[str, Any]: Data on cuda.
        """
        if self._to_cuda:
            for key, data_item in data.items():
                if isinstance(data_item, torch.Tensor):
                    if self._pin_memory:
                        data_item = data_item.pin_memory()
                    if self._use_channel_last:
                        if len(data_item.shape) == 4:
                            data[key] = data_item.cuda(non_blocking=True).to(  # type: ignore[call-overload] # pylint: disable=line-too-long # noqa: E501
                                memory_format=torch.channels_last
                            )
                        else:
                            data[key] = data_item.cuda(non_blocking=True)
                    else:
                        data[key] = data_item.cuda(non_blocking=True)

                    del data_item
        elif self._recursive_to_cuda:
            data = self.recursive_to_cuda(data)

        return data

    def get_batch_data(self) -> Dict[str, Any]:
        """Get one batch data.

        Returns:
            One batch Data. Dict[str, Any].
        """
        try:
            data = next(self._data_iter)
        except StopIteration:
            self.reset()
            data = next(self._data_iter)

        if self._to_cuda or self._recursive_to_cuda:
            if not isinstance(data, dict):
                raise ValueError(f"Expect type(data)=dict, but got {type(data)}.")

        data = self.move_data_to_cuda(data)

        self._step += 1
        return data


def _init_worker_funcion(
    workers_per_gpu: int = 1,
    seed: Optional[int] = None,
) -> Any:
    """Initialize worker function.

    Args:
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU, generally as random seed.
        seed (int, Optional): Seed to be used. Default: None.

    Returns:
        Callable function.
    """
    rank = (
        get_data_parallel_rank() if dist.is_available() and dist.is_initialized() else 0
    )

    init_fn = (
        partial(_worker_init_fn, num_workers=workers_per_gpu, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    return init_fn


def _worker_init_fn(
    worker_id: int,
    num_workers: int,
    rank: int,
    seed: int,
) -> None:
    """Worker init function.

    Args:
        worker_id (int): The worker id.
        num_workers (int): Worker number.
        rank (int): The process rank.
        seed (int): Seed to be used.
    """
    affinity = os.sched_getaffinity(0)
    logger.warning(
        "GPU Process %d DataLoaderWorker %d set affinity to: %s",
        rank,
        worker_id,
        affinity,
    )
    # The seed of each worker equals to (num_worker * rank + worker_id + user_seed)
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -*- coding: utf-8 -*-
"""UniRobot Samplers.

Pay attention to docstring under URDistributedSampler/Warning.

We have been ensure determinstic:
1. The indices numbers are exactly the same when the epoch settings are the same.
2. When the epoch setting is different, the indices is also different.
3. The data of each batch is only related to batch_size, and this determinism can be
   guaranteed under different numbers of GPUs.
"""

import logging
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import Sampler

from unirobot.brain.data.dataset.base_dataset import BaseDataset
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_world_size,
)


logger = logging.getLogger(__name__)


class URDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such a case, each process can pass
    a `unirobot.brain.data.sampler.URDistributedSampler` instance as a
    `unirobot.brain.data.dataloader.URDataLoader` sampler, and load a subset of the original
    dataset that is exclusive to it.

    Args:
        dataset: Dataset used for sampling.
        shuffle (bool): If `True` (default), sampler will shuffle the
            indices.
        seed (int): random seed used to shuffle the sampler if
            `shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: `0`.
        drop_last (bool): If `True`, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If `False`, the sampler will add extra indices to make
            the data evenly divisible across the replicas.
            Default: `False`.
        indices (list, optional): If set, will use this instead of the default
            `list(range(len(self.dataset)))`.
            Default: `None`.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, `world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within `num_replicas`.
            By default, `rank` is retrieved from the current distributed
            group.
        batch_size (int, optional): How many samples per batch to load
            (default: ``1``).

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example:
        >>> sampler = URDistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-branches
        self,
        dataset: BaseDataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        indices: Optional[List[int]] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        batch_size: int = 1,
    ):
        """URDistributedSampler."""
        if indices is not None and not isinstance(indices, list):
            raise ValueError(f"Expect indices list of int, but got {indices}.")
        if not isinstance(dataset, BaseDataset):
            raise ValueError(
                f"Expect type(dataset)=BaseDataset but got {type(dataset)}."
            )
        if indices is not None:
            logger.warning(
                "Weighted sample will not be be effective because of setting "
                "indices manually."
            )
        self._dataset = dataset
        self._epoch = 0
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._seed = seed
        self._indices = indices
        self._step = 0
        self._batch_size = batch_size

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = get_data_parallel_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = get_data_parallel_rank()
        if rank not in range(0, num_replicas):
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval"
                f" [0, {num_replicas - 1}].",
            )

        self._num_replicas = num_replicas
        self._rank = rank

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        self._index_repeat = self._dataset.get_index_repeat()
        if len(self._index_repeat) != len(self._dataset):
            raise ValueError(
                "_index_repeat size != _dataset size. Details"
                f"{len(self._index_repeat)} != {len(self._dataset)}"
            )
        if (
            np.all(np.array(self._index_repeat) == np.array(self._index_repeat)[0])
            and self._index_repeat[0] == 1
        ):
            self.actual_data_size = len(
                np.repeat(
                    np.array(list(range(len(self._dataset)))), self._index_repeat
                ).tolist()
            )

        else:
            if self._batch_size > 1:
                self.actual_data_size = len(self._dataset)
                # self.actual_data_size = (
                #     len(self._dataset) - len(self._dataset) % self._batch_size
                # )
                # self.actual_data_size = len(
                #     np.repeat(
                #         np.array(list(range(self.actual_data_size))),
                #         self._index_repeat[: self.actual_data_size],
                #     ).tolist()
                # )
            else:
                self.actual_data_size = len(
                    np.repeat(
                        np.array(list(range(len(self._dataset)))), self._index_repeat
                    ).tolist()
                )

        sample_len = len(indices) if indices else self.actual_data_size
        if self._drop_last and sample_len % self._num_replicas != 0:
            self._num_samples = math.ceil(
                (sample_len - self._num_replicas) / self._num_replicas,
            )
        else:
            self._num_samples = math.ceil(sample_len / self._num_replicas)
        self._total_size = self._num_samples * self._num_replicas

    def indices_sampling_per_rank_hook(  # pylint: disable=no-self-use.
        self, indices
    ) -> Union[List[Dict[str, int]], List[int]]:
        """Rank-wise sampling hook.

        Args:
            indices (Union[List[Dict[str, int]], List[int]]): The indices.

        Returns:
            Union[List[Dict[str, int]], List[int]]: The indices.
        """
        return indices

    def _get_sampler_size(self) -> int:
        """Rebuild indices based index repeat.

        Returns:
            indices (List): The return indices of rebuilding.
        """
        if (
            np.all(np.array(self._index_repeat) == np.array(self._index_repeat)[0])
            and self._index_repeat[0] == 1
        ):
            pass
        else:
            if self._batch_size > 1:
                return int(self._num_samples * self._index_repeat[0])
        return self._num_samples

    def _check_num_samples(self, indices: List[int]) -> None:
        """Rebuild indices based index repeat.

        Returns:
            indices (List): The return indices of rebuilding.
        """
        if (
            np.all(np.array(self._index_repeat) == np.array(self._index_repeat)[0])
            and self._index_repeat[0] == 1
        ):
            if len(indices) != self._num_samples:
                raise RuntimeError(
                    f"Expect len(indices) == self._num_samples, but got "
                    f"len(indices)={len(indices)}, "
                    f"self.num_samples={self._num_samples}",
                )
        else:
            if self._batch_size > 1:
                pass
            else:
                if len(indices) != self._num_samples:
                    raise RuntimeError(
                        f"Expect len(indices) == self._num_samples, but got "
                        f"len(indices)={len(indices)}, "
                        f"self.num_samples={self._num_samples}",
                    )

    def _rebuild_rank_indices(
        self, indices: List[Union[int, Any]]
    ) -> List[Union[int, Any]]:
        """Rebuild indices based index repeat.

        Returns:
            indices (List): The return indices of rebuilding.
        """
        if (
            np.all(np.array(self._index_repeat) == np.array(self._index_repeat)[0])
            and self._index_repeat[0] == 1
        ):
            pass
        else:
            if self._batch_size > 1:
                if len(indices) < self._batch_size:
                    raise ValueError(
                        f"rank: {self._rank},indices: ({indices}) "
                        f"size: {len(indices)} < batch size: {self._batch_size}"
                    )
                indices = indices[: len(indices) - len(indices) % self._batch_size]
                if len(indices) == 0:
                    return []
                np_indices_0 = np.array(indices)
                np_indices = np.split(np_indices_0, len(indices) // self._batch_size)

                if not np.all(
                    np.array(self._index_repeat) == np.array(self._index_repeat)[0]
                ):
                    raise ValueError(
                        "When batch size > 1, you need align temporal,"
                        "So _index_repeat should have same value."
                        f"{self._index_repeat}"
                    )
                np_indices = np.repeat(np_indices, self._index_repeat[0], axis=0)  # type: ignore[assignment]
                indices = np_indices.flatten().tolist()  # type: ignore[attr-defined]

                indices = [
                    (
                        inds,
                        (idx // self._batch_size) % self._index_repeat[0],
                        not (idx // self._batch_size) % self._index_repeat[0] == 0,
                    )
                    for idx, inds in enumerate(indices)
                ]
            else:
                clip_id, frame_idx, _ = indices[0]  # type: ignore[misc]
                indices[0] = (clip_id, frame_idx, False)
        return indices

    def _rebuild_indices(self, indices: List[Union[int, Any]]) -> List[Union[int, Any]]:
        """Rebuild indices based index repeat.

        Returns:
            indices (List): The return indices of rebuilding.
        """
        if (
            np.all(np.array(self._index_repeat) == np.array(self._index_repeat)[0])
            and self._index_repeat[0] == 1
        ):
            index_repeat_array = np.array(self._index_repeat)[indices]
            indices = np.repeat(np.array(indices), index_repeat_array).tolist()
        else:
            if self._batch_size > 1:
                if not np.all(
                    np.array(self._index_repeat) == np.array(self._index_repeat)[0]
                ):
                    raise ValueError(
                        "When batch size > 1, you need align temporal,"
                        "So _index_repeat should have same value."
                        f"{self._index_repeat}"
                    )
                return indices

            index_repeat_array = np.array(self._index_repeat)[indices]
            frame_idx_per_clip = []
            for repeat_num in index_repeat_array.tolist():
                frame_idx_list = []
                for i in range(repeat_num):
                    if i == 0:
                        frame_idx_list.append([i, False])
                    else:
                        frame_idx_list.append([i, True])
                frame_idx_per_clip.extend(frame_idx_list)

            indices = np.repeat(np.array(indices), index_repeat_array).tolist()
            indices = [
                (inds, frame_idx_per_clip[idx][0], frame_idx_per_clip[idx][1])
                for idx, inds in enumerate(indices)
            ]
            del frame_idx_per_clip
        return indices

    def _generate_indices(self) -> Union[List[int], List[Dict[str, int]]]:
        """Generate indices.

        Returns:
            If self._indices is not None, return self._indices, else return
            Union[List[int], List[Dict[str, int]]].
        """
        if self._indices is not None:
            if self._shuffle:
                np.random.seed(seed=self._epoch + self._seed)
                np.random.shuffle(self._indices)
                indices = self._indices
            else:
                indices = self._indices
        else:
            if not self._shuffle:
                logger.warning(
                    "Weighted sampling indices is disabled because of "
                    "self._shuffle=False."
                )
                indices = list(range(len(self._dataset)))
                indices = self._rebuild_indices(indices)
            else:
                shuffle_seed = self._seed + self._epoch
                dataset_size = len(self._dataset)
                sample_weight = self._dataset.get_sample_weight()
                if len(sample_weight) != len(self._dataset):
                    raise ValueError(
                        f"Expect len(sample_weight)==len(self._dataset) but got "
                        f"len(sample_weight)={len(sample_weight)}, "
                        f"len(self._dataset)={len(self._dataset)}."
                    )
                same_weight = len(set(sample_weight)) == 1

                if hasattr(self._dataset, "get_indices"):
                    indices = self._dataset.get_indices(
                        shuffle_seed,
                        len(self._dataset),
                    )
                elif dataset_size >= (2**24):
                    # Note: torch would raise
                    # `RuntimeError: number of categories cannot exceed 2^24` if
                    # dataset_size >= (2**24), this is a known issues on github, more
                    # details at `https://github.com/pytorch/pytorch/issues/2576`
                    if not same_weight:
                        raise RuntimeError(
                            "Weighted sample is not supported if dataset size >= "
                            "(2**24). Note: torch would raise "
                            "`RuntimeError: number of categories cannot exceed 2^24` "
                            "if dataset_size >= (2**24), this is a known issues on "
                            "github, more details at "
                            "`https://github.com/pytorch/pytorch/issues/2576`"
                        )
                    np.random.seed(shuffle_seed)
                    indices = np.random.permutation(dataset_size).tolist()
                else:
                    # deterministically shuffle based on epoch and seed
                    t_generator = torch.Generator()
                    t_generator.manual_seed(shuffle_seed)
                    # Sampling without replacement if sampling weight is all
                    # the same.
                    replacement = not same_weight
                    indices = torch.multinomial(
                        torch.tensor(sample_weight).float(),
                        dataset_size,
                        replacement=replacement,
                        generator=t_generator,
                    ).tolist()
                    indices = self._rebuild_indices(indices)

        return indices

    def __iter__(self):
        """Overwrite iter."""
        indices = self._generate_indices()

        if not self._drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self._total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self._total_size]

        if len(indices) != self._total_size:
            raise RuntimeError(
                f"Expect len(indices) == self.total_size, but got "
                f"len(indices)={len(indices)}, "
                f"self.total_size={self._total_size}.",
            )

        # subsample
        offset = self._num_samples * self._rank
        indices = indices[offset : offset + self._num_samples]

        indices = self._rebuild_rank_indices(indices)

        self._check_num_samples(indices)

        # if len(indices) != self._num_samples:
        #     raise RuntimeError(
        #         f"Expect len(indices) == self._num_samples, but got "
        #         f"len(indices)={len(indices)}, "
        #         f"self.num_samples={self._num_samples}",
        #     )

        indices = indices[self._step * self._batch_size :]

        indices = self.indices_sampling_per_rank_hook(indices)
        self._step = 0

        return iter(indices)

    def __len__(self) -> int:
        """Return length of sampler, i.e num samples."""
        return self._get_sampler_size()
        # return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different random
        ordering for each epoch. Otherwise, the next iteration of this sampler will
        yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError(
                f"Expect epoch is a non-negative integer integer, bug got {epoch}."
            )

        self._epoch = epoch

    def get_epoch(self) -> int:
        """Get the epoch."""
        return self._epoch

    def set_step(self, step: int) -> None:
        """Set the step for this sampler.

        Args:
            step (int): Step number.
        """
        self._step = step

    def set_indices(self, indices: List[int]) -> None:
        """
        Set the indices for this sampler. It's useful to update indices while training.

        Args:
            indices (int): Epoch number.
        """
        if indices is not None and not isinstance(indices, list):
            raise ValueError(f"Expect indices list of int, but got {indices}.")

        self._indices = indices

    def get_indices(self) -> Optional[List[int]]:
        """Get the indices."""
        return self._indices

    @property
    def drop_last(self):
        """Return drop last."""
        return self._drop_last

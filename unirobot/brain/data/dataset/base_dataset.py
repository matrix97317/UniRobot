# -*- coding: utf-8 -*-
"""UniRobot BaseDataset."""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose

from unirobot.utils.unirobot_slot import TRANSFORM


logger = logging.getLogger(__name__)


class BaseDataset(Dataset[T_co], ABC):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Different from torch, we recommend all subclass
    overwrite :meth:`__len__`, which is expected to return the size of the dataset by
    many :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    Args:
        mode (str): Mode in _VALID_MODE.
        meta_file (str or None) : Meta file, expect Dict[str, Dict[str, List[str]]],
        transforms (list, dict or None): List of transforms.

    .. code-block:: python

        >>> meta_file={
        ...     'lidar'={
        ...         'train'=['train_meta_file1', 'train_meta_file1'...],
        ...         'val'=['val_meta_file1', 'val_meta_file1'...],
        ...         'test'=['test_meta_file1', 'test_meta_file1'...],
        ...     },
        ...     'camera'={...},
        ...     ...
        ... }.

    .. note::
        :class:`~torch.utils.data.DataLoader` by default constructs a index
        sampler that yields integral indices.  To make it work with a map-style
        dataset with non-integral indices/keys, a custom sampler must be provided.

    """

    _VALID_MODE = (
        "train",
        "val",
        "test",
    )

    def __init__(
        self,
        mode: str,
        meta_file: Optional[Dict[str, Dict[str, List[str]]]] = None,
        transforms: Union[List[Any], Dict[str, Any], None] = None,
    ) -> None:
        """Init."""
        if mode not in self._VALID_MODE:
            logger.warning(
                "Expect mode in %s, but got %s.",
                str(self._VALID_MODE),
                mode,
            )

        self._mode = mode
        self._meta_file = meta_file
        if transforms is None:
            self._transforms = None
        elif isinstance(transforms, dict):
            self._transforms = TRANSFORM.build(transforms)
        elif isinstance(transforms, list):
            _transforms = []
            for transform in transforms:
                _transforms.append(TRANSFORM.build(transform))
            self._transforms = Compose(_transforms)
        else:
            raise ValueError(
                f"Expect `type(transforms)` in (list, dict), if transforms is not "
                f"None, but got {type(transforms)}."
            )

    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Get one data item."""
        raise NotImplementedError()

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        """Concat Datasets."""
        return ConcatDataset([self, other])

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        raise NotImplementedError()

    @staticmethod
    def batch_collate(data: List[Any]) -> Any:
        """Batch collate function.

        Args:
            data (List): List of `__getitem__` returns.
        """
        return default_collate(data)

    def get_sample_weight(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get sample weight.

        Returns:
            List of float or {str, List of float} pairs: each element in list
            corresponds to the sampling weight of the current index element.
            Default same probability for each sample.

        """
        return [1.0] * len(self)

    def get_index_repeat(self) -> List[int]:
        """Get repeat value for per index.

        Returns:
            List of float or {str, List of float} pairs: The return list contains
            repeat times per index of the sample.

        """
        return [1] * len(self)

    def get_labeling_task_ids(self) -> List[str]:  # pylint: disable=no-self-use
        """Get labeling task ids.

        The default return is empty list. And it should be overridden by subclasses.
        Return empty list now to avoid breaking incompatible training tasks.

        Returns:
            List of str: each element is a labeling task id.
        """
        return []

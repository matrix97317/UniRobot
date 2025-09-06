# -*- coding: utf-8 -*-
"""UniRobot BaseTransform."""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torchvision
from PIL import Image

from unirobot.utils.unirobot_slot import TRANSFORM


class BaseTransform(ABC):  # pylint: disable=too-few-public-methods
    """Tranform Abstract Interface.

    Args:
        use_gpu (bool): Whether to use gpu.
            Default=`False`.
    """

    def __init__(
        self,
        use_gpu: bool = False,
    ) -> None:
        """Initialize."""
        self._use_gpu = use_gpu

    @abstractmethod
    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement of transform."""
        raise NotImplementedError()


class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (List[Union[Callable, Dict[str, Any]]]): Sequence of transform
            object or config dict to be composed.
        use_gpu (bool): Whether to use gpu.
            Default=`False`.
    """

    def __init__(
        self,
        transforms: List[Union[Callable, Dict[str, Any]]],
        use_gpu: bool = False,
    ):
        """Initialize compose operation."""
        super().__init__(use_gpu=use_gpu)

        if not isinstance(transforms, list):
            raise ValueError(
                f"Expect type(trainsforms)=list, but got {type(transforms)}."
            )

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                self.transforms.append(TRANSFORM.build(transform))
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("Transform must be callable or Dict.")

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Call function to apply transforms sequentially.

        Args:
            data_dict (Dict[str, Any]): A dict contains the data to transform.

        Returns:
           Transformed data.
        """
        if not data_dict:
            raise ValueError("Data is None or empty.")

        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict

    def __repr__(self) -> str:
        """Show all transform operations.

        Returns:
            A string contains all transforms.
        """
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += f"    {transform}"
        format_string += "\n)"
        return format_string


class ToTorchTensor(BaseTransform):  # pylint: disable=too-few-public-methods
    """Convert PIL Image and np.ndarray to torch.Tensor.

    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].

    Args:
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(self, use_gpu: bool = False):
        """Initialize."""
        super().__init__(use_gpu=use_gpu)

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to tensor.

        Args:
            data_dict (Dict[str, torch.Tensor]): A dict contains the data to
                torch.Tensor.

        Returns:
           torch.Tensor.
        """
        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, Image.Image)):
                data_dict[key] = torchvision.transforms.ToTensor()(value)
        return data_dict

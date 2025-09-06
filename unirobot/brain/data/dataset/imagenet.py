# -*- coding: utf-8 -*-
"""Dataset of ImageNet."""

import io
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from PIL import Image

from unirobot.brain.data.dataset.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


class ImageNetDataset(BaseDataset):
    """Dataset for ImageNet data.

    Args:
        mode (str): Mode in _VALID_MODE.
        meta_file (Dict[str, Dict[str, List[str]]] or None) : Meta file.
        meta_file_key (str or None) : Meta file key, expect str.

    """

    DATASET_META_FILE = {
        "imagenet": {
            "train": ["imagenet_train_new.txt"],
            "val": ["imagenet_val_new.txt"],
        },
    }

    def __init__(
        self,
        mode: str,
        meta_file: Optional[Dict[str, Dict[str, List[str]]]] = None,
        meta_file_key: str = "imagenet",
        transforms: Union[List[Any], Dict[str, Any], None] = None,
    ) -> None:
        """Init."""
        super().__init__(
            mode=mode,
            transforms=transforms,
        )
        meta_file = meta_file or self.DATASET_META_FILE
        self._meta_file = meta_file[meta_file_key]  # type: ignore[assignment]
        self._data_paths = self._meta_file[self._mode]  # type: ignore[index]
        # self._cache_records = self.parse_meta_file()
        logger.info("Data path: %s.", self._data_paths)
        

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, int]]:
        """Get one item.

        Args:
            idx (int): The index.

        Returns:
            One data item, which contains `image` and `gt`.
        """
        # data_path, gt = self._cache_records[idx]
        # image = self.data_reader(data_path)
        image = np.random.randint(0, 256, size=(600,400,3), dtype=np.uint8)
        gt = np.random.randint(0,1000,size=(1,))[0]

        data_dict = {
            "image": Image.fromarray(image),
            "gt": gt,
        }

        if self._transforms is not None:
            data_dict = self._transforms(data_dict)

        return data_dict

    def parse_meta_file(self) -> List[Tuple[str, int]]:
        """Parse data path.

        Returns:
            cache_records (list): List of {hash_code, gt} for each data.
        """
        cache_records: List[Tuple[str, int]] = []
        for data_path in self._data_paths:
            with open(data_path, encoding="utf8") as fin:
                for line in fin:
                    (
                        path,
                        gt,
                        hash_code,
                        file_size,
                    ) = line.split()
                    del path
                    cache_records.append((hash_code + " " + file_size, int(gt)))
        return cache_records

    def __len__(self) -> int:
        """Get length."""
        return 500000

    # @staticmethod
    # def data_reader(data_path: str) -> Image.Image:
    #     """Read image.

    #     Args:
    #         data_path (str): Hash code of image.

    #     Returns:
    #         RGB image object(PIL Image.Image).
    #     """
    #     image_bytes = read_bytes([data_path])[0]
    #     with Image.open(io.BytesIO(image_bytes)) as image:
    #         image = image.convert("RGB")
    #     return image


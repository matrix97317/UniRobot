# -*- coding: utf-8 -*-
"""2D data transform function."""
import logging
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from unirobot.brain.data.transform.base_transform import BaseTransform


logger = logging.getLogger(__name__)


class MultiScaleCrop(BaseTransform):
    """MultiScaleCrop.

    Args:
        output_size (int|list|tuple): Output size after crop on the original image.
        scales (int|List): Scale size of original image.
        max_distort (int): Allowable image distort size.
        fix_crop (bool): Use fixed crop offset to crop image.
        more_fix_crop (bool): Use more fixed crop offset to crop image.
        use_gpu (bool): Whether to use gpu. Default=`False`.

    Returns:
        inputs['image']: (output_size,output_size,3)
    """

    def __init__(
        self,
        output_size: Union[int, List[int], Tuple[int, int]],
        scales: Optional[List[float]] = None,
        max_distort: int = 1,
        fix_crop: bool = True,
        more_fix_crop: bool = True,
        use_gpu: bool = False,
    ):
        """Init MultiScaleCrop."""
        super().__init__(use_gpu=use_gpu)
        self._scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self._max_distort = max_distort
        self._fix_crop = fix_crop
        self._more_fix_crop = more_fix_crop
        self._output_size = (
            output_size
            if not isinstance(output_size, int)
            else [output_size, output_size]
        )
        self._interpolation = Image.Resampling.BILINEAR

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return Crop Image.

        Args:
            data_dict (Dict[str, Any]): A dict contains the data to transform.

        Returns:
           Transformed data.
        """
        input_img = data_dict["image"]
        input_img_size = input_img.size
        # calculate crop size and crop position
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(input_img_size)
        crop_image = input_img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)
        )
        # resize to output size
        resize_image = crop_image.resize(
            (self._output_size[0], self._output_size[1]), self._interpolation
        )

        data_dict["image"] = resize_image
        return data_dict

    def _sample_crop_size(
        self,
        im_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """Return Crop Size and Crop Position.

        Args:
            im_size (Tuple[int, int]): Image shape size.

        Returns:
            Tuple[int, int, int, int]. i.e. crop_pair[0], crop_pair[1], w_offset,
                h_offset
        """
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self._scales]
        crop_h = [
            self._output_size[1] if abs(x - self._output_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self._output_size[0] if abs(x - self._output_size[0]) < 3 else x
            for x in crop_sizes
        ]
        pairs = []
        for i, height in enumerate(crop_h):
            for j, width in enumerate(crop_w):
                if abs(i - j) <= self._max_distort:
                    pairs.append((height, width))
        # select specific scale to crop
        crop_pair = random.choice(pairs)
        # calculate different crop position
        if not self._fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(
        self,
        image_w: int,
        image_h: int,
        crop_w: int,
        crop_h: int,
    ) -> Tuple[int, int]:
        """Return offset which is used to crop.

        Args:
            image_w (int): Width of image.
            image_h (int): Height of image.
            crop_w (int): Width of image to crop.
            crop_h (int): Height of image to crop.

        Returns:
            Offset which is used to crop.
        """
        offsets = self.fill_fix_offset(
            self._more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(
        more_fix_crop: bool,
        image_w: int,
        image_h: int,
        crop_w: int,
        crop_h: int,
    ) -> List:
        """Calculate offset by origin image size and crop size.

        Args:
            image_w (int): Width of image.
            image_h (int): Height of image.
            crop_w (int): Width of image to crop.
            crop_h (int): Height of image to crop.
        """
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = []
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower right quarter

        return ret

    @staticmethod
    def fill_fc_fix_offset(
        image_w: int,
        image_h: int,
        crop_w: int,
        crop_h: int,
    ) -> List:
        """Calculate offset by origin image size and crop size.

        Args:
            image_w (int): Width of image.
            image_h (int): Height of image.
            crop_w (int): Width of image to crop.
            crop_h (int): Height of image to crop.
        """
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2

        ret = []
        ret.append((0, 0))  # left
        ret.append((1 * w_step, 1 * h_step))  # center
        ret.append((2 * w_step, 2 * h_step))  # right

        return ret


class RandomHorizontalFlip(BaseTransform):  # pylint: disable=too-few-public-methods
    """Randomly horizontally flips.

    The given PIL.Image with a probability of 0.5

    Args:
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Init RandomHorizontalFlip."""
        super().__init__(use_gpu=use_gpu)

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call function to flip images.

        Args:
            data_dict (Dict[str, Any]): A dict contains the data to transform.

        Returns:
           Transformed data.
        """
        image = data_dict["image"]
        prob = random.random()
        if prob < 0.5:
            flip_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            data_dict["image"] = flip_image
            return data_dict
        return data_dict


class Normalize(BaseTransform):  # pylint: disable=too-few-public-methods
    """Normalize the image.

    Args:
        mean (List[float]): List of mean values.
        std (List[float]): List of std values.
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        use_gpu: bool = False,
    ) -> None:
        """Init Normalize."""
        super().__init__(use_gpu=use_gpu)
        self._mean = mean
        self._std = std

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call function to normalize images.

        Args:
            data_dict (Dict[str, Any]): Result dict from loading pipeline.

        Returns:
            data_dict['image']: it's value = (x-mean)/std
        """
        image = data_dict["image"]
        # TODO: This should be separated to a standalone # pylint:disable=fixme.
        # Transform, i.e. ToFloatTransform.
        if image.dtype in (torch.half, torch.float, torch.double):
            pass
        elif image.dtype == torch.int:
            image = image.float()
        elif image.dtype == torch.int64:
            image = image.double()
        elif image.dtype == torch.int16:
            image = image.half()
        else:
            logger.warning("Unsupported torch.dtype: %s.", image.dtype)
            image = image.float()
        rep_mean = self._mean * (image.size()[0] // len(self._mean))
        rep_std = self._std * (image.size()[0] // len(self._std))

        for tensor, mean, std in zip(image, rep_mean, rep_std):
            tensor.sub_(mean).div_(std)
        data_dict["image"] = image
        return data_dict


class Scale(BaseTransform):  # pylint: disable=too-few-public-methods
    """Scale.

    Args:
        scale_size (int): The scale size.
        interpolation (Any): The interpolation.
        use_gpu (bool): Whether to use gpu. Default=`False`.


    Rescales the input PIL.Image to the given 'size'. 'size' will be the size of the
    smaller edge. For example, if height > width, then image will be rescaled to
    (size * height / width, size) size: size of the smaller edge interpolation:
    Default=`torchvision.transforms.functional.InterpolationMode.BILINEAR`.
    """

    def __init__(
        self,
        scale_size: int,
        interpolation: Any = InterpolationMode.BILINEAR,
        use_gpu: bool = False,
    ) -> None:
        """Init Scale."""
        super().__init__(use_gpu=use_gpu)
        self._scale_size = scale_size
        self._interpolation = interpolation

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call function to scaling images.

        Args:
            data_dict (Dict[str, Any]): Result dict from loading pipeline.

        Returns:
            data_dict['image']:(scale_size * height / width, scale_size,3)
        """
        image = data_dict["image"]
        scale_image = torchvision.transforms.Resize(
            self._scale_size, self._interpolation
        )(image)
        data_dict["image"] = scale_image
        return data_dict


class CenterCrop(BaseTransform):  # pylint: disable=too-few-public-methods
    """CenterCrop.

    Args:
        output_size (int): The output size.
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(
        self,
        output_size: int,
        use_gpu: bool = False,
    ) -> None:
        """Init CenterCrop."""
        super().__init__(use_gpu=use_gpu)
        self._output_size = output_size

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call function to crop images in center.

        Args:
            data_dict (Dict[str, Any]): Result dict from loading pipeline.

        Returns:
            data_dict['image']:(output_size,output_size,3)
        """
        image = data_dict["image"]
        crop_image = torchvision.transforms.CenterCrop(self._output_size)(image)
        data_dict["image"] = crop_image
        return data_dict


class GroupFCSample(BaseTransform):  # pylint: disable=too-few-public-methods
    """Full Crop for Sample.

    Args:
        output_size (int): The output size.
        scale_size (int, optional): The scale size.
        use_gpu (bool): Whether to use gpu. Default=`False`.
    """

    def __init__(
        self,
        output_size: int,
        scale_size: Optional[int] = None,
        use_gpu: bool = False,
    ):
        """Init full crop."""
        super().__init__(use_gpu=use_gpu)

        self._output_size = (
            output_size
            if not isinstance(output_size, int)
            else (output_size, output_size)
        )
        self._scale_worker = None
        if scale_size is not None:
            self._scale_worker = Scale(scale_size)

    def __call__(
        self,
        data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Full Crop image.

        Args:
            data_dict (Dict[str, Any]): Result dict from loading pipeline.

        Returns:
            data_dict['image']: [img1,img2,img3]
        """
        if self._scale_worker is not None:
            data_dict = self._scale_worker(data_dict)

        image = data_dict["image"]
        image_w, image_h = image.size

        offsets = MultiScaleCrop.fill_fc_fix_offset(image_w, image_h, image_h, image_h)
        oversample_group = []
        for o_w, o_h in offsets:
            crop_image = image.crop((o_w, o_h, o_w + image_h, o_h + image_h))
            oversample_group.append(crop_image)
        data_dict["image"] = oversample_group
        return data_dict

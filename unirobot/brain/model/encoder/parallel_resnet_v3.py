# -*- coding: utf-8 -*-
"""Provide all resnet model."""

import logging
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import torch
from torch import nn


from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_rank,
)


logger = logging.getLogger(__name__)


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        dilation (int, optional): Spacing between kernel elements. Default: 1

    Returns:
        3x3 conv instance.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """1x1 convolution.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1

    Returns:
        1x1 conv instance.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """BasicBlock.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        downsample (nn.Module, optional): Downsample module.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        base_width (int, optional): Middle channel. Default: 64.
        dilation (int, optional): Spacing between kernel elements. Default: 1
        norm_layer (nn.Module, optional): Normalize layer, e.g. nn.BatchNorm2d.
            Default=`None`.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs,
    ) -> None:
        """Init BasicBlock."""
        super().__init__(**kwargs)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        downsample (nn.Module, optional): Downsample module.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        base_width (int, optional): Middle channel. Default: 64.
        dilation (int, optional): Spacing between kernel elements.
            Default: 1
        norm_layer (nn.Module, optional): Normalize layer, e.g. nn.BatchNorm2d.
            Default=`None`.

    Bottleneck in torchvision places the stride for downsampling at 3x3
    convolution(self.conv2). While original implementation places the stride at the
    first 1x1 convolution(self.conv1). According to "Deep residual learning for image
    recognition "https://arxiv.org/abs/1512.03385. This variant is also known as
    ResNet V1.5 and improves accuracy according to https://ngc.nvidia.com/catalog/
    model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """Initialize Bottleneck."""
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        identity = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet.

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): Base block.
        layers (List[int]): Number of blocks for each layer.
        num_classes (int, optional): The number of categories that the network
            finally outputs. Default=1000.
        zero_init_residual (bool, optional): Whether to init residual by zero.
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        width_per_group (int, optional): Middle channel. Default: 64.
        replace_stride_with_dilation (List[bool], optional): Whether to replace
            stride with dilation. Default: None.
        norm_layer (Callable[..., nn.Module] or None): Normalize layer,
            e.g. nn.BatchNorm2d. Default: `None`.
    """

    def __init__(  # pylint: disable=too-many-branches
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        pipeline_parallel_rank: int = 0,
    ) -> None:
        """Initialize Resnet."""
        super().__init__()
        # Pipeline Parallel Params
        self.pipeline_parallel_rank = pipeline_parallel_rank
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        logger.warning("ResNet Use %s .", self._norm_layer)

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}",
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.pipeline_parallel_rank == 0:
            self.conv1 = nn.Conv2d(
                3,
                self.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.bn1 = norm_layer(self.in_channels)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.in_channels = 64
            self.layer1 = self._make_layer(block, 64, layers[0])

            # if self.pipeline_parallel_rank == 1:
            self.in_channels = 256
            self.layer2 = self._make_layer(
                block,
                128,
                layers[1],
                stride=2,
                dilate=replace_stride_with_dilation[0],
            )

            # if self.pipeline_parallel_rank == 2:
            self.in_channels = 512
            self.layer3 = self._make_layer(
                block,
                256,
                layers[2],
                stride=2,
                dilate=replace_stride_with_dilation[1],
            )
            # if self.pipeline_parallel_rank == 3:
            self.in_channels = 1024
            self.layer4 = self._make_layer(
                block,
                512,
                layers[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.full_catenate = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual
        # branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Make layer.

        Args:
            block (Type[Union[BasicBlock, Bottleneck]]): Base block.
            out_channels (int): Number of channels produced by the convolution
            blocks (List[int]): Number of blocks.
            stride (int, optional): Stride of the convolution. Default: 1
            dilate (bool, optional): Whether to dilate. Default: False.

        Returns:
            nn.Sequential of blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ),
            )

        return nn.Sequential(*layers)

    def _forward_impl(
        self,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        # See note [TorchScript super()]

        if self.pipeline_parallel_rank == 0:
            out = self.conv1(inputs[0])
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

            out = self.layer1(out)

            # out = self.layer2(out)
            # if self.pipeline_parallel_rank == 1:
            out = self.layer2(out)

            # out = self.layer3(out)

            # if self.pipeline_parallel_rank == 2:
            out = self.layer3(out)

            # out = self.layer4(out)
            # out = self.avgpool(out)
            # out = torch.flatten(out, 1)
            # out = self.full_catenate(out)

            # if self.pipeline_parallel_rank == 3:
            out = self.layer4(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.full_catenate(out)

        return [out]

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self._forward_impl(inputs)


class ResNet18(torch.nn.Module):
    """ResNet18.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize Resnet18."""
        super().__init__()
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    def init_weight(self) -> None:
        """Initialize resnet18 weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class ResNet34(torch.nn.Module):
    """Resnet34.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init resnet34."""
        super().__init__()

        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    def init_weight(self) -> None:
        """Init weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class ParallelResNet50V3(torch.nn.Module):
    """Resnet50.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        use_sync_bn: bool = False,
        **kwargs,
    ) -> None:
        """Init resnet50."""
        super().__init__()
        self._pretrain_model = pretrain_model or None
        if use_sync_bn:
            kwargs["norm_layer"] = nn.SyncBatchNorm
        kwargs["pipeline_parallel_rank"] = get_pipeline_model_parallel_rank()
        self.base_model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def init_weight(self) -> None:
        """Init resnet50 weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def forward(self,inputs: torch.Tensor)->torch.Tensor:
        return self.train_forward(inputs)


class ResNet101(torch.nn.Module):
    """Resnet101.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init resnet101."""
        super().__init__()
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    def init_weight(self) -> None:
        """Init resnet101 weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class ResNet152(torch.nn.Module):
    """Resnet152.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init resnet152."""
        super().__init__()
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    def init_weight(self) -> None:
        """Init weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class ResNet50W4G32(torch.nn.Module):
    """Resnet50_32x4d.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init Resnet50_32x4d."""
        super().__init__()
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 4
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def init_weight(self) -> None:
        """Init Resnet50_32x4d weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class ResNet101G8W32(torch.nn.Module):
    """ResNet101_32x8d.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init ResNet101_32x8d."""
        super().__init__()
        kwargs["groups"] = 32
        kwargs["width_per_group"] = 8
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    def init_weight(self) -> None:
        """Init ResNet101_32x8d weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class WideResNet50W128(torch.nn.Module):
    """WideResNet50_2.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init WideResNet50_2."""
        super().__init__()
        kwargs["width_per_group"] = 64 * 2
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    def init_weight(self) -> None:
        """Init WideResNet50_2 weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)


class WideResNet101W128(torch.nn.Module):
    """WideResNet101_2.

    Args:
        train_mode (bool): Whether to enable train mode.
        pretrain_model (str): Path to pretrain model. Default: None.
    """

    def __init__(
        self,
        train_mode: bool = True,
        pretrain_model: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Init WideResNet101_2."""
        super().__init__()
        kwargs["width_per_group"] = 64 * 2
        self._pretrain_model = pretrain_model or None
        self.base_model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    def init_weight(self) -> None:
        """Init WideResNet101_2 weight."""
        if self._pretrain_model is not None:
            # place holder: load checkpoint.
            pass

    def infer_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Infer base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

    def train_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Train base model forward.

        Args:
            inputs (torch.Tensor): Forward inputs.

        Returns:
            Forward results (torch.Tensor).
        """
        return self.base_model(inputs)

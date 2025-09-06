# -*- coding: utf-8 -*-
"""A Simple Trainer."""
import itertools
import json
import logging
import os
import random
import statistics
import time
from collections import OrderedDict
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
from prettytable import PrettyTable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from torch.optim import Optimizer
from tqdm import tqdm

from unirobot.brain.data.dataloader.base_dataloader import URDataLoader
from unirobot.brain.evaluator.base_evaluator import BaseEvaluator
from unirobot.brain.model.base_model_flow import ModelFlow
from unirobot.brain.model.optimizer.base_lr_scheduler import BaseLrScheduler
from unirobot.brain.model.optimizer.base_optimizer import build_optimizer
from unirobot.brain.trainer.base_trainer import BaseTrainer
from unirobot.utils import constants
from unirobot.utils.cfg_parser import PyConfig
from unirobot.brain.infra.checkpoint_util import CheckpointUtil
from unirobot.utils.file_util import FileUtil
from unirobot.utils.json_encoder import URJsonEncoder
from unirobot.brain.infra.distributed.data_parallel.wrap_model import (
    DistributedDataParallel as LocalDDP,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_group,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_data_parallel_world_size,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_model_parallel_group,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_pipeline_model_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    get_tensor_model_parallel_rank,
)
from unirobot.brain.infra.distributed.initialization.parallel_state import (
    initialize_model_parallel,
)
from unirobot.brain.infra.distributed.optimizer.optimizer import FP32Optimizer
from unirobot.brain.infra.distributed.pipeline_parallel.pp_scheme_v2 import (
    forward_backward_no_pipelining,
)
from unirobot.brain.infra.distributed.pipeline_parallel.pp_scheme_v2 import (
    forward_backward_pipelining_with_1F1B,
)
from unirobot.utils.unirobot_slot import DATALOADER
from unirobot.utils.unirobot_slot import EVALUATOR
from unirobot.utils.unirobot_slot import LR_SCHEDULER
from unirobot.utils.unirobot_slot import MODEL_FLOW
from unirobot.brain.infra.stopwatch import ProfileStopwatch
from unirobot.brain.infra.stopwatch import Stopwatch
from unirobot.brain.infra.tensorboard_util import TensorboardUtil


if LooseVersion(torch.__version__) >= LooseVersion("2.0.0"):
    import functools

    from torch.distributed.fsdp import (  # pylint: disable=import-error,no-name-in-module,ungrouped-imports,line-too-long # noqa: E501
        FullyShardedDataParallel,
    )
    from torch.distributed.fsdp.fully_sharded_data_parallel import (  # pylint: disable=import-error,no-name-in-module,line-too-long # noqa: E501
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import (  # pylint: disable=import-error,no-name-in-module,line-too-long # noqa: E501
        size_based_auto_wrap_policy,
    )


logger = logging.getLogger(__name__)


class ParallelTrainer(BaseTrainer):  # pylint: disable=too-many-instance-attributes
    """ParallelTrainer provides 3-D Parallel Training Policy.

    Args:
        cfg (PyConfig): cfg_path is experiment config file path.
        run_name (str): run_name is tag of experiment, that is used to distinguish
            different experiments.
        resume (bool, optional): Experiments run name. Default=`False`.
        total_epochs (int, optional): Training epochs. Default=`1`.
        print_freq_by_step (int, optional): Print frequency. Default=`20`.
        save_ckpt_freq_by_epoch (int, optional): Save checkpoint frequency of epoch.
            Default=`None`.
        save_ckpt_freq_by_step (int, optional): Save checkpoint frequency of step.
            Default=`None`.
        enable_sync_loss (bool, optional): Whether to use sync loss. Note: Using rank0
            loss if this set to false. Default=`True`.
        ues_fp16 (bool, optional): Whether to enable fp16 forward. Default=`False`.
        use_sync_bn (bool, optional): Whether to use sync bn. Note: disable could
            accelerate, but may cost reduce the accuracy. Default=`False`.
        clip_gradient (int, optional): If gradient value bigger than clip_gradient, clip
            to `clip_gradient`. Default=`35`.
        lr_per_sample (int, optional): Learning rate per sample. Default=`1e-4`.
        use_deterministic (bool, optional): Whether to enable deterministic algorithms
            of pytorch. Default=`False`.
        use_tensorboard (bool): Whether to enable tensorboard, the output will
            save into output/exp_xxx/your_experiments_name/summary. Default=`False`.
        use_grad_none (bool): Whether to set None in optimizer.zero_grad() and
            model.zero_grad(). Default=`False`.
        use_model_channel_last (bool): Whether to enable channel last in model.
            This trick will make full use of TensorCore. It is worth noting
            that operators need to be supported. Default=`False`.
        use_cudnn_benchmark (bool): Whether to use cudnn benchmark accelerate pipeline.
            Default=`False`.
        use_data_stream (bool): Whether to use data stream to asynchronous Memcpy(H2D).
        grad_accumulate_step (bool): Set gradient accumulate's step times. Default=1.
        amp_scaler_params (dict): The params of AutoMixedPrecision grad scaler.
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.
        ddp_params (dict): The params of DistributedDataParallel.
            https://pytorch.org/docs/1.9.0/_modules/torch/nn/parallel/distributed.html
        ddp_type: Set data parallel method. Default=local.
        megatron_optimizer_params (Dict): The config params of Megatron.
        data_parallel_random_init (bool): In local DDP, use random init model.
        accumulate_allreduce_grads_in_fp32 (bool): In local DDP, use float32 data type.
        use_contiguous_buffers_in_local_ddp (bool): In local DDP, use contiguous buffer.
        tensor_parallel_size (int): Set the number of gpu for a group tensor parallel.
            Default=1.
        pipeline_parallel_size (int): Set the number of gpu for a group
            pipeline parallel. Default=1.
        seed (int, optional): Seed for deterministic. Default=`666`.
        log_precision (int, optional): Decimal precision. Default=`4`.
        barrier_before_stopwatch (bool): Whether to barrier process before stopwatch.
            If 'True', time cost is more accurate when using data parallel.
            Default=`False`.
        pure_dataloading (bool): Whether to use pure dataloading. If `True`,
            just loading data without training model. Use pure dtaloading and pure model
            at the same time is not allowed. Default=`False`.
        pure_model (bool): Whether to use pure model. If `True`, training model
            with a fixed batch data.  Use pure dtaloading and pure model at the same
            time is not allowed. Default=`False`.
        skip_exception (bool): Whether to skip exception. If `True`, continue next step
            training if occurs error, else abort training. Default=`False`.
        use_eval_mode (bool): Eval Model Mode. Default=`False`.
        eval_ckpt_list (bool): The ckpt list for eval model.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-statements,too-many-locals,too-many-branches # noqa: E501
        self,
        cfg: PyConfig,
        run_name: str,
        resume: bool = False,
        total_epochs: int = 1,
        print_freq_by_step: int = 20,
        save_ckpt_freq_by_epoch: Optional[int] = None,
        save_ckpt_freq_by_step: Optional[int] = None,
        max_save_ckpt_num: Optional[int] = None,
        enable_sync_loss: bool = True,
        use_fp16: bool = False,
        use_model_half: bool = False,
        use_cpu_offload: bool = False,
        use_sync_bn: bool = False,
        clip_gradient: int = 35,
        lr_per_sample: float = 1e-3,
        use_deterministic: bool = False,
        use_tensorboard: bool = False,
        use_log_grad: bool = False,
        use_grad_none: bool = False,
        use_model_channel_last: bool = False,
        use_cudnn_benchmark: bool = False,
        use_data_stream: bool = False,
        grad_accumulate_step: int = 1,
        amp_scaler_params: Optional[Dict] = None,
        ddp_params: Optional[Dict] = None,
        ddp_type: str = "local",
        use_fsdp: bool = False,
        use_nsight: bool = False,
        nsight_warmup_iters: int = 10,
        megatron_optimizer_params: Optional[Dict] = None,
        data_parallel_random_init: bool = True,
        accumulate_allreduce_grads_in_fp32: bool = True,
        use_contiguous_buffers_in_local_ddp: bool = True,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        num_micro_batch: Optional[int] = None,
        seed: int = constants.seed,
        log_precision: int = 4,
        barrier_before_stopwatch: bool = False,
        pure_dataloading: bool = False,
        pure_model: bool = False,
        skip_exception: bool = False,
        use_eval_mode: bool = False,
        eval_ckpt_list: Optional[list] = None,
        use_interval: bool = False,
        infer_type: Optional[str] = None,
        export_type: Optional[str] = None,
    ) -> None:
        """Init BaseTrainer based config dict."""
        super().__init__(cfg)
        torch.cuda.empty_cache()

        self._run_name = run_name
        self._resume = resume
        self._func_hook_list: dict = {}
        # Nsight Params
        self._use_nsight = use_nsight
        self._nsight_warmup_iters = nsight_warmup_iters

        # init trainer paramters
        self._infer_type = infer_type
        self._export_type = export_type
        self._total_epochs = total_epochs
        self._epoch_idx = 0
        self._start_epoch = 0
        self._one_epoch_size = 1
        self._step_idx = 0
        self._start_step = 0
        self._time_anchor = 0.0
        self._barrier_before_stopwatch = barrier_before_stopwatch
        self._pure_dataloading = pure_dataloading
        self._pure_model = pure_model
        self._batch_data: Optional[Dict[str, Any]] = None
        self._use_interval = use_interval
        if self._pure_dataloading and self._pure_model:
            raise ValueError(
                "pure_dataloading and pure_model can't be `true` at the same time."
            )

        # placeholders mostly for logging.
        self._time_cost: DefaultDict[str, List[float]] = defaultdict(list)
        self._log_vars: OrderedDict[Any, Any] = OrderedDict()

        self._print_freq_by_step = print_freq_by_step
        self._save_ckpt_freq_by_epoch = save_ckpt_freq_by_epoch
        self._save_ckpt_freq_by_step = save_ckpt_freq_by_step
        self._max_save_ckpt_num = max_save_ckpt_num
        if (
            self._save_ckpt_freq_by_epoch is None
            and self._save_ckpt_freq_by_step is None
        ):
            raise ValueError(
                "you must set save_ckpt_freq_by_epoch or save_ckpt_freq_by_step!!!"
            )
        if self._save_ckpt_freq_by_epoch is not None:
            self._save_ckpt_freq_by_step = None

        self._use_fp16 = use_fp16
        self._use_model_half = use_model_half
        self._use_cpu_offload = use_cpu_offload
        if self._use_cpu_offload:
            logger.warning("============> Use CPU Offload Mode. <=============")
        self._use_sync_bn = use_sync_bn
        self._enable_sync_loss = enable_sync_loss
        if enable_sync_loss:
            logger.warning("Enable sync loss to log.")
        else:
            logger.warning(
                "Disable sync loss to log. This will acclerate train_parse_loss."
            )
        self._clip_gradient = clip_gradient
        self._lr_per_sample = lr_per_sample
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

        self._rank = dist.get_rank()
        self._gpu_num = dist.get_world_size()
        self._total_norm: torch.Tensor = torch.Tensor([0.0])
        self._best_eval = 0
        self._log_prec = log_precision
        # enable enable_deterministic
        self._seed = seed
        self._use_deterministic = use_deterministic
        # enable tensorboard
        self._use_tensorboard = use_tensorboard
        self._use_log_grad = use_log_grad
        if self._use_tensorboard:
            self._summary = TensorboardUtil(
                log_dir=os.path.join(FileUtil.get_summary_dir(), f"rank{self._rank}"),
                comment=self._run_name,
                rank=self._rank,
            )
        # will grad set as None
        self._use_grad_none = use_grad_none
        # enable model channel last
        self._use_model_channel_last = use_model_channel_last
        # enable CUDNN benchmark
        self._use_cudnn_benchmark = use_cudnn_benchmark
        # check gradient accumulate step
        self._gradient_accumulate_step = grad_accumulate_step
        if self._gradient_accumulate_step < 1:
            raise ValueError("grad_accumulate_step must be >=1 ")
        logger.warning(
            "=====> Use Gradient Accumulate Step: %d <======",
            self._gradient_accumulate_step,
        )
        # check amp_scaler_params
        self._amp_scaler_params = amp_scaler_params

        # check ddp_params
        self._ddp_params = ddp_params
        self._ddp_static_graph = False
        self._ddp_type = ddp_type
        self._use_fsdp = use_fsdp
        self._data_parallel_random_init = data_parallel_random_init
        self._accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
        self._use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        # enable data stream
        self._use_data_stream = use_data_stream
        if self._use_data_stream:
            self._device: Any = None
            self._data_stream: Optional[torch.cuda.Stream] = None
            self._wrap_dataloader: Any = None
            self._first: bool = True
            self._next_data: Any = None

        # parallel params
        self._tensor_parallel_size = tensor_parallel_size
        self._pipeline_parallel_size = pipeline_parallel_size
        self._num_micro_batch = pipeline_parallel_size
        if num_micro_batch is not None:
            self._num_micro_batch = num_micro_batch

        if self._use_fsdp:
            self._num_micro_batch = 1

        # Megatron Optimizer Params
        self._megatron_optimizer_params = {
            "clip_grad": self._clip_gradient,
            "log_num_zeros_in_grad": False,
        }
        if megatron_optimizer_params is not None:
            self._megatron_optimizer_params.update(megatron_optimizer_params)

        # Eval Mode
        self._use_eval_mode = use_eval_mode
        self._eval_ckpt_list = eval_ckpt_list
        self._ckpt_path: str = ""

        self._init_parallel_state()
        self._use_data_parallel = False
        if tensor_parallel_size == 1 and pipeline_parallel_size == 1:
            self._use_data_parallel = True
            logger.warning("====> Currently, Use Data Parallel.")
        else:
            ddp_size = self._gpu_num // (tensor_parallel_size * pipeline_parallel_size)
            logger.warning(
                "====> Currently, Use 3D Parallel, "
                "Data Parallel Groups %s , "
                "Pipeline Parallel Stages %s ,"
                "Tensor Parallel Width %s",
                ddp_size,
                pipeline_parallel_size,
                tensor_parallel_size,
            )

        self.enable_deterministic()

        # Init Trainer Control Params
        self._build_model_flow()
        self._build_performance_components()
        self._build_dataloader()
        self._build_optimizer()
        if not self._use_eval_mode:
            self._build_lr_scheduler()
            self._build_ckpt_manager()
        else:
            self._build_evaluator()
            self.load_model_file()

        # Create stopwatch
        self._stopwatch = Stopwatch(False)

        self._skip_exception = skip_exception
        # Profiler
        self._profiler = ProfileStopwatch(
            "profiler", sync_before_stopwatch=False, barrier_before_stopwatch=False
        )

    def enable_deterministic(self) -> None:
        """Enable system deterministic."""
        torch.use_deterministic_algorithms(self._use_deterministic)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)

    def _init_parallel_state(self) -> None:
        """Init parallel state."""
        initialize_model_parallel(
            self._tensor_parallel_size,
            self._pipeline_parallel_size,
        )

    def _build_dataloader(self) -> None:
        """Build dataloader."""
        if self._use_eval_mode:
            self._dataloader = DATALOADER.build(self._cfg.infer_dataloader)
        else:
            self._dataloader = DATALOADER.build(self._cfg.dataloader)
        if not isinstance(self._dataloader, URDataLoader):
            raise ValueError(
                f"Expect type(self._dataloader)=URDataLoader, "
                f"but got {type(self._dataloader)}."
            )
        self._sample_num_per_gpu = self._dataloader.get_sample_num_per_gpu()
        self._sample_per_epoch = self._dataloader.get_sample_per_epoch()
        if self._pipeline_parallel_size > 1:
            self._one_epoch_size = (
                self._dataloader.get_one_epoch_step_per_gpu() // self._num_micro_batch
            )
        else:
            self._one_epoch_size = self._dataloader.get_one_epoch_step_per_gpu()

        self._total_steps = self._total_epochs * self._one_epoch_size
        self._bs_per_gpu = self._dataloader.get_batch_size_per_gpu()
        if self._pipeline_parallel_size > 1:
            self._bs_per_gpu = self._bs_per_gpu * self._num_micro_batch  # type: ignore[operator] # noqa: E501 # pylint: disable=line-too-long

    def _build_performance_components(self) -> None:
        """Build performance components."""
        # Enable fp16.
        if self._use_fp16:
            actual_amp_scaler_params = {}
            if self._amp_scaler_params is not None:
                if "init_scale" in self._amp_scaler_params:
                    actual_amp_scaler_params["init_scale"] = self._amp_scaler_params[
                        "init_scale"
                    ]

                if "growth_factor" in self._amp_scaler_params:
                    actual_amp_scaler_params["growth_factor"] = self._amp_scaler_params[
                        "growth_factor"
                    ]

                if "backoff_factor" in self._amp_scaler_params:
                    actual_amp_scaler_params["backoff_factor"] = (
                        self._amp_scaler_params["backoff_factor"]
                    )

                if "growth_interval" in self._amp_scaler_params:
                    actual_amp_scaler_params["growth_interval"] = (
                        self._amp_scaler_params["growth_interval"]
                    )

                logger.warning(
                    "You set amp_scaler_params as %s", actual_amp_scaler_params
                )
            actual_amp_scaler_params["enabled"] = self._use_fp16
            self._scaler = torch.cuda.amp.GradScaler(**actual_amp_scaler_params)
            logger.warning("======> Use AMP FP16 <=========")

    def _build_model_flow(self) -> None:  # pylint: disable=too-many-branches
        """Build model flow."""
        if self._use_eval_mode:
            self._model_flow = MODEL_FLOW.build(self._cfg.infer_model_flow)
        else:
            self._model_flow = MODEL_FLOW.build(self._cfg.model_flow)

        self._full_model = (
            self._model_flow._full_model  # pylint: disable=protected-access
        )
        self._loss_func = (
            self._model_flow._loss_func
        )  # pylint: disable=protected-access

        if not isinstance(self._model_flow, ModelFlow):
            raise ValueError(
                f"Expect type(self._model)=ModelFlow, but got {type(self._model_flow)}."
            )
        if not isinstance(self._full_model, list):
            self._full_model = [self._full_model]

        if get_data_parallel_rank() == 0:
            model_module_params_sum = []
            for model_module in self._full_model:
                layer_params_sum = []
                for params in model_module.parameters():
                    layer_params_sum.append(params.nelement())
                model_module_params_sum.append(sum(layer_params_sum))
            logger.warning(
                " > number of parameters on (tensor, pipeline) "
                "model parallel rank (%s, %s): %.2f",
                get_tensor_model_parallel_rank(),
                get_pipeline_model_parallel_rank(),
                sum(model_module_params_sum),
            )
        # convert model to float16
        if self._use_model_half:
            logger.warning("============> Use model half <===========")
            for model_module in self._full_model:
                model_module.to(torch.float16)

        # GPU allocation.
        for model_module in self._full_model:
            model_module.cuda(torch.cuda.current_device())

        # enable model channel last
        if self._use_model_channel_last:
            logger.warning("============> Use model_channel_last <===========")
            for model_module in self._full_model:
                model_module.to(memory_format=torch.channels_last)

        gpus_cur_machine = torch.cuda.device_count()
        device_ids = list(range(0, gpus_cur_machine))
        logger.info(
            "[%s] rank = %d, world_size = %d, n = %d, device_ids = %s.",
            os.getpid(),
            dist.get_rank(),
            dist.get_world_size(),
            gpus_cur_machine,
            str(device_ids),
        )
        # DDP Model
        if self._ddp_type == "torch":
            if self._use_fsdp:
                logger.warning("============> Use FSDP. <=============")
                my_auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=20000
                )
                self._full_model = [
                    FullyShardedDataParallel(
                        model_module,
                        process_group=get_data_parallel_group(),
                        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                        auto_wrap_policy=my_auto_wrap_policy,
                    )
                    for model_module in self._full_model
                ]
            else:
                logger.warning("============> Use DDP. <=============")
                i = torch.cuda.current_device()
                self._full_model = [
                    torchDDP(
                        model_module,
                        device_ids=[i],
                        output_device=i,
                        process_group=get_data_parallel_group(),
                    )
                    for model_module in self._full_model
                ]

        elif self._ddp_type == "local":
            self._full_model = [
                LocalDDP(
                    model_module,
                    self._accumulate_allreduce_grads_in_fp32,
                    self._use_contiguous_buffers_in_local_ddp,
                )
                for model_module in self._full_model
            ]
            # broad cast params from data parallel src rank to other data parallel ranks
            if self._data_parallel_random_init:
                for model_module in self._full_model:
                    model_module.broadcast_params()
        else:
            raise NotImplementedError(
                f"Unknown DDP implementation specified: " f"{self._ddp_type}. Exiting."
            )

    def _build_optimizer(self) -> None:
        """Build optimizer."""
        if not self._use_fsdp:
            model_params_policy = self._model_flow.get_optim_policies()
            self._cfg.optimizer.params_policy = model_params_policy
            self._cfg.optimizer.lr = (
                self._lr_per_sample * get_data_parallel_world_size() * self._bs_per_gpu
            )

            optimizer = build_optimizer(**(self._cfg.optimizer))
            if not isinstance(optimizer, Optimizer):
                raise ValueError(
                    f"Expect type(self._optimizer)=Optimizer, "
                    f"but got {type(optimizer)}."
                )

            for group in model_params_policy:
                logger.info(
                    "group: %s has %d params, lr_mult: %f, decay_mult: %f.",
                    group["name"],
                    len(group["params"]),
                    group["lr_mult"],
                    group["decay_mult"],
                )
        else:
            self._cfg.optimizer.params_policy = self._full_model[0].parameters()
            self._cfg.optimizer.lr = (
                self._lr_per_sample * get_data_parallel_world_size() * self._bs_per_gpu
            )
            self._cfg.optimizer.use_fsdp = True
            optimizer = build_optimizer(**(self._cfg.optimizer))
        params_have_main_grad = False
        if self._ddp_type == "local":
            params_have_main_grad = True
        self._optimizer = FP32Optimizer(
            optimizer,
            self._megatron_optimizer_params["clip_grad"],
            self._megatron_optimizer_params["log_num_zeros_in_grad"],
            params_have_main_grad,
            self._use_contiguous_buffers_in_local_ddp,
            self._ddp_type,
            self._full_model,
            use_fsdp=self._use_fsdp,
            scaler=self._scaler,
        )

    def _build_lr_scheduler(self) -> None:
        """Build lr scheduler."""
        self._cfg.lr_scheduler.optimizer = self._optimizer
        self._lr_scheduler = LR_SCHEDULER.build(self._cfg.lr_scheduler)
        if not isinstance(self._lr_scheduler, BaseLrScheduler):
            raise ValueError(
                f"Expect type(self._lr_scheduler)=BaseLrScheduler, "
                f"but got {type(self._lr_scheduler)}."
            )

    def _build_evaluator(self) -> None:
        """Build evaluator."""
        self._evaluator = EVALUATOR.build(self._cfg.evaluator)
        if not isinstance(self._evaluator, BaseEvaluator):
            raise ValueError(
                f"Expect type(self._evaluator)=BaseEvaluator, "
                f"but got {type(self._evaluator)}."
            )

    def load_model_file(
        self,
    ) -> None:
        """Load ckpt or resume ckpt.

        Args:
            ckpt_path (str): Path to checkpoint.
        """
        ckpt_prefix = f"checkpoint_pipeline_rank_{get_pipeline_model_parallel_rank()}"
        for ckpt in self._eval_ckpt_list:
            if ckpt_prefix in ckpt:
                self._ckpt_path = ckpt
                break
        if self._ckpt_path is None:
            raise ValueError(f" {self._eval_ckpt_list} have not {ckpt_prefix} .")

        self._ckpt_manager = CheckpointUtil(
            model=self._full_model[0],
            optimizer=None,
            lr_scheduler=None,
            scaler=None,
            save_dir=None,
            pretrain_model=self._ckpt_path,
            resume=False,
            to_cuda=self._cfg.infer_ckpt.to_cuda,
            ckpt2model_json=self._cfg.infer_ckpt.ckpt2model_json,
            trace_mode=False,
            use_interval=self._use_interval,
        )
        self._ckpt_manager.load_model()

    def _build_ckpt_manager(self) -> None:
        """Build ckpt manager."""
        ckpt_cfg = self._cfg.ckpt
        ckpt_cfg.update(
            dict(
                model=self._full_model[0],
                optimizer=self._optimizer,
                lr_scheduler=self._lr_scheduler,
                scaler=self._scaler,
                save_dir=FileUtil.get_ckpt_dir(),
                resume=self._resume,
                use_interval=self._use_interval,
            )
        )
        self._ckpt_manager = CheckpointUtil(**ckpt_cfg)

        # load ckpt
        self.load_ckpt()

    def load_ckpt(self) -> None:
        """Auto load ckpt."""
        self._ckpt_manager.load_model()
        self._ckpt_manager.load_lr_scheduler()
        self._ckpt_manager.load_optimizer()
        self._ckpt_manager.load_scaler()
        self._start_epoch = self._ckpt_manager.load_epoch()
        self._start_step = self._ckpt_manager.load_step()
        self._step_idx = self._start_step
        logger.warning(
            "==> Rank %d, Starting Training Epoch %d Step %d.",
            self._rank,
            self._start_epoch,
            self._start_step,
        )

    def save_eval_results(
        self,
        save_path: str,
        eval_results: Dict,
    ) -> Any:
        """Save evaluation's results.

        Args:
            save_path (str): Path to save eval results.
            eval_results (Any): Eval results to save.
        """
        if self._rank == self._gpu_num - 1:
            try:
                json_str = json.dumps(
                    eval_results, indent=4, sort_keys=True, cls=URJsonEncoder
                )
            except TypeError as ex:
                raise TypeError(
                    "The `eval_results` should be composed of dict or list "
                    "without tensor or ndarray."
                ) from ex
            with open(save_path, "a", encoding="utf8") as json_file:
                json_file.write(json_str)

    def infer(self, *args, **kwargs):
        """Inferrer Process."""
        self.train(*args, **kwargs)

    def train(self) -> None:  # pylint: disable=too-many-branches
        """Training Process."""
        self._time_anchor = time.time()
        try:
            if self._use_eval_mode:
                with torch.no_grad():
                    # eval mode.
                    self._full_model[0].eval()
                    self._evaluator.set_zero()
                    self._dataloader.set_state(epoch=0, step=0, indices=None)
                    eval_results = self.infer_one_epoch()
                    dist.barrier(group=get_model_parallel_group())
                    if eval_results is not None:
                        save_path = os.path.join(
                            FileUtil.get_export_dir(),
                            f"{os.path.basename(self._ckpt_path)}_eval_result.json",
                        )
                        logger.info(
                            "Saving evaluate results of checkpoint %s to %s.\n"
                            "Evaluate result: %s.",
                            self._ckpt_path,
                            save_path,
                            eval_results,
                        )

                        self.save_eval_results(save_path, eval_results)
                    else:
                        logger.warning("The return of infer_one_epoch() is None.")

            else:
                torch.backends.cudnn.benchmark = self._use_cudnn_benchmark
                self._lr_scheduler.init_base_lr()
                self._full_model[0].train()
                for self._epoch_idx in range(self._start_epoch, self._total_epochs):
                    if "custom_indices_generator" in self._func_hook_list:
                        indices = self._func_hook_list["custom_indices_generator"]()
                    else:
                        indices = None

                    # Set state to ensure data indices different at each epoch.
                    self._dataloader.set_state(
                        epoch=self._epoch_idx, step=self._step_idx, indices=indices
                    )
                    self.train_one_epoch(self._epoch_idx)
                    if self._save_ckpt_freq_by_epoch is not None:
                        save_by_freq = (
                            self._epoch_idx % self._save_ckpt_freq_by_epoch == 0
                        )
                        if not self._use_fsdp:
                            self._ckpt_manager.save_checkpoint(
                                epoch=self._epoch_idx + 1,
                                step=self._step_idx,
                                max_save_ckpt_num=self._max_save_ckpt_num,
                                save_by_freq=save_by_freq,
                            )

                # save last epoch ckpt when needed.
                if (
                    self._save_ckpt_freq_by_epoch is not None
                    and self._epoch_idx % self._save_ckpt_freq_by_epoch != 0
                ):
                    if not self._use_fsdp:
                        self._ckpt_manager.save_checkpoint(
                            epoch=self._epoch_idx + 1,
                            step=self._step_idx,
                        )

                if self._use_fsdp:
                    self._ckpt_manager.save_checkpoint(
                        epoch=self._epoch_idx + 1,
                        step=self._step_idx,
                    )

                # close tensorboard
                if self._use_tensorboard:
                    self._summary.close()

        except Exception as ex:  # pylint: disable=broad-except
            dist.barrier()
            time.sleep(20)
            dist.barrier()
            ex_message = ex.args[0] if len(ex.args) > 0 else ""
            logger.error(
                "Training occurs error, aborting this experiments. Will save all state "
                "info...{%s}",
                ex_message,
            )
            # save loader state
            self._ckpt_manager.save_checkpoint(
                epoch=self._epoch_idx,
                step=self._step_idx,
            )
            # close tensorboard
            if self._use_tensorboard:
                self._summary.close()

            raise ex

    def infer_one_epoch(self) -> Any:
        """One epoch infering."""
        eval_results = None
        for self._step_idx in tqdm(range(0, self._one_epoch_size)):
            eval_results = self.infer_one_step()

            if self._step_idx % self._print_freq_by_step == 0:
                logger.info(self)  # see more details at `self.__str__()`.
                self._time_cost = defaultdict(list)

        if self._step_idx % self._print_freq_by_step != 0:
            logger.info(self)
            self._time_cost = defaultdict(list)
        if eval_results is not None:
            return self._evaluator.dist_all_reduce(eval_results)
        return None

    def infer_one_step(self) -> Any:
        """One step infering.

        Returns:
            One step eval results.

        """
        # Forward pass.
        eval_results = []
        self._stopwatch.tic()
        if self._use_data_parallel:
            outputs = forward_backward_no_pipelining(
                self._dataloader,
                self._loss_func,
                self._full_model,
                self._optimizer,
                True,
                True,
                profiler=self._profiler,
                num_microbatches=self._num_micro_batch,
                amp_scaler=self._scaler,
                use_cpu_offload=self._use_cpu_offload,
            )
        else:
            outputs = forward_backward_pipelining_with_1F1B(
                self._dataloader,
                self._loss_func,
                self._full_model,
                self._optimizer,
                self._cfg,
                True,
                True,
                profiler=self._profiler,
                num_microbatches=self._num_micro_batch,
                amp_scaler=self._scaler,
                use_cpu_offload=self._use_cpu_offload,
            )

        self.record_time_cost("forward_backward", "default")
        # print(outputs)
        if len(outputs) > 0:
            for output in outputs:
                res = self._evaluator.eval(output[0], output[1])
                eval_results.append(res)
        # print(eval_results)
        self.record_time_cost("eval_cost")
        if len(eval_results) > 0:
            return eval_results[-1]
        return None

    def train_one_epoch(
        self,
        epoch: int,
    ) -> None:
        """One epoch training.

        Args:
            epoch (int): Current epoch.
        """
        self._time_cost = defaultdict(list)
        for self._step_idx in range(self._start_step, self._one_epoch_size):
            # if self.skip_exception, skip this one step when occuring error,
            # else raise error and aborting this experiments
            try:
                outputs = self.train_one_step(epoch, self._step_idx)
            except Exception as ex:  # pylint: disable=broad-except
                logger.exception("Found error in step %d", self._step_idx)
                if not self._skip_exception:
                    raise ex
            else:
                # rely on outputs
                if "custom_sampler_indices" in self._func_hook_list:
                    indices = self._func_hook_list["custom_sampler_indices"](
                        self._log_vars, outputs
                    )
                    self._dataloader.set_state(
                        epoch=epoch, step=self._step_idx, indices=indices
                    )
            if self._step_idx % self._print_freq_by_step == 0:
                logger.info(self)  # see more details at `self.__str__()`.
                self._time_cost = defaultdict(list)
                self._profiler.reset()

            if self._save_ckpt_freq_by_step is not None:
                if self._step_idx % self._save_ckpt_freq_by_step == 0:
                    if not self._use_fsdp:
                        self._ckpt_manager.save_checkpoint(
                            epoch=epoch,
                            step=self._step_idx + 1,
                            max_save_ckpt_num=self._max_save_ckpt_num,
                            save_by_freq=True,
                        )

        # save last step ckpt when needed.
        if (
            self._save_ckpt_freq_by_step is not None
            and self._step_idx % self._save_ckpt_freq_by_step != 0
        ):
            if not self._use_fsdp:
                self._ckpt_manager.save_checkpoint(
                    epoch=self._epoch_idx + 1,
                    step=self._step_idx,
                )

        if self._step_idx % self._print_freq_by_step != 0:
            logger.info(self)
            self._time_cost = defaultdict(list)
            self._profiler.reset()

        # reset start_step
        self._start_step = 0
        self._step_idx = 0

    def get_batch_data(self) -> Dict[str, Any]:
        """Get batch data.

        Returns:
            One batch data (Dict[str, Any]).
        """
        batch_data = self._dataloader.get_batch_data()
        return batch_data

    def train_one_step(  # pylint: disable=too-many-branches,too-many-statements
        self,
        epoch: int,
        step: int,
    ) -> Union[Dict[str, Any], torch.Tensor, Dict[str, Dict[str, Any]]]:
        """One step training.

        Args:
            epoch (int): Current epoch.
            step (int): Current step.

        Returns:
            Model outputs (Union[Dict[str, torch.Tensor], torch.Tensor]) and
            time cost (Dict[str, float]).
        """
        # Set grad to zero.
        self._stopwatch.tic()
        if self._ddp_type == "local" and self._use_contiguous_buffers_in_local_ddp:
            for partition in self._full_model:
                partition.zero_grad_buffer()
        self._optimizer.zero_grad()

        if self._use_nsight:
            if step == self._nsight_warmup_iters:
                torch.cuda.cudart().cudaProfilerStart()
                logger.warning("Rank %s: start nsight profiling.", self._rank)

            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_push(f"rank_{self._rank}_iteration_{step}")

            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_push(f"rank_{self._rank}_forward_backward")

        # Forward pass.
        # self._stopwatch.tic()
        if self._use_data_parallel:
            outputs = forward_backward_no_pipelining(
                self._dataloader,
                self._loss_func,
                self._full_model,
                self._optimizer,
                profiler=self._profiler,
                num_microbatches=self._num_micro_batch,
                amp_scaler=self._scaler,
                use_cpu_offload=self._use_cpu_offload,
            )
        else:
            outputs = forward_backward_pipelining_with_1F1B(
                self._dataloader,
                self._loss_func,
                self._full_model,
                self._optimizer,
                self._cfg,
                profiler=self._profiler,
                num_microbatches=self._num_micro_batch,
                amp_scaler=self._scaler,
                use_cpu_offload=self._use_cpu_offload,
            )

        for output in outputs:
            if isinstance(output, dict):
                self._log_vars["default"] = output

        if self._use_nsight:
            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_pop()

        if self._use_nsight:
            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_push(f"rank_{self._rank}_optimizer")

        # Reduce gradients.
        self._optimizer.reduce_model_grads()

        # Update parameters.
        update_successful, grad_norm, _ = self._optimizer.step()
        self._total_norm = grad_norm
        # Gather params.
        if update_successful:
            self._optimizer.gather_model_params()

        if self._use_nsight:
            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_pop()

        if self._use_nsight:
            if step >= self._nsight_warmup_iters:
                torch.cuda.nvtx.range_pop()

        if self._use_nsight:
            if step == (self._nsight_warmup_iters + 10):
                torch.cuda.cudart().cudaProfilerStop()
                logger.warning("==> Rank %s: stop nsight profiling.", self._rank)

        # Update learning rate.
        # lr_scheduler step
        if hasattr(self._lr_scheduler, "step"):
            self._lr_scheduler.step(
                epoch * self._one_epoch_size + step,
                epoch,
            )
        self.record_time_cost("forward_backward", "default")
        return outputs

    def record_time_cost(self, time_cost_key: str, time_cost_key_prefix: str = ""):
        """Record time cost.

        Args:
            time_cost_key (str): Key of time cost.
            time_cost_key_prefix (str): Time cost key prefix. Default=`""`.
        """
        if self._barrier_before_stopwatch:
            torch.cuda.synchronize()
            dist.barrier()

        if time_cost_key_prefix != "":
            time_cost_key_prefix += "/"

        self._time_cost[f"{time_cost_key_prefix}{time_cost_key}"].append(
            self._stopwatch.toc2()
        )

    def register_hook(
        self,
        func_name: str,
        func: Callable,
    ) -> None:
        """Register custom function, that is inserted into training flow.

        Args:
            func_name (str): Register hook name.
            func (Callable): Register hook func.
        """
        if not callable(func):
            raise ValueError(
                f"Expect callable(func)=True, but got False. func: {func}."
            )

        self._func_hook_list[func_name] = func

    def get_model(self) -> Any:
        """Return model is used to training."""
        return self._full_model

    def get_optimizer(self) -> Any:
        """Return optimizer is used to training."""
        return self._optimizer

    def get_lr_scheduler(self) -> Any:
        """Return lr scheduler is used to training."""
        return self._lr_scheduler

    def get_dataloader(self) -> Any:
        """Return dataloader is used to training."""
        return self._dataloader

    def __str__(self) -> str:
        """Represent trainer info as table."""
        table = PrettyTable()
        table.field_names = ["Basic Info", "Loss", "Time Cost"]
        table.align["Loss"] = "r"
        table.align["Time Cost"] = "r"

        # loss
        loss_info_list = []
        for task_name, task_loss in self._log_vars.items():
            if task_name != "":
                loss_info_list.append(f"Task: {task_name}")
                loss_prefix = task_name + "/"
            else:
                loss_prefix = ""
            loss_info_list.append(
                f"{loss_prefix}loss: {task_loss['loss']:.{self._log_prec}f}"
            )
            loss_info_list.extend(
                [
                    f"{loss_prefix}{loss_name}: {loss_value:.{self._log_prec}f}"
                    for loss_name, loss_value in task_loss.items()
                    if loss_name != "loss"
                ]
            )

        # time_cost
        one_step_time_cost = 0.0
        time_cost: List[str] = []
        world_size = dist.get_world_size() if dist.is_available() else 1
        for item_name, item_value in self._time_cost.items():
            time_cost.append(
                f"{item_name}: {statistics.mean(item_value):.{self._log_prec}f}"
            )
            one_step_time_cost += statistics.mean(item_value)

        # profiler
        for item_name, item_value in self._profiler._time_cost.items():
            time_cost.append(
                f"{item_name}: {statistics.mean(item_value):.{self._log_prec}f}"
            )

        # compute training consumed and eta time
        time_consumed = timedelta(seconds=int(time.time() - self._time_anchor))
        global_step = self._epoch_idx * self._one_epoch_size + self._step_idx
        eta_training_seconds = one_step_time_cost * (self._total_steps - global_step)
        eta_training = timedelta(seconds=int(eta_training_seconds))
        # ETA of current epoch
        eta_curr_epoch_seconds = one_step_time_cost * (
            self._one_epoch_size - global_step % self._one_epoch_size
        )
        eta_curr_epoch = timedelta(seconds=int(eta_curr_epoch_seconds))
        if self._pipeline_parallel_size > 1:
            one_step_samples = self._bs_per_gpu * get_data_parallel_world_size()
        else:
            one_step_samples = self._bs_per_gpu * world_size  # type: ignore[operator]

        if one_step_time_cost > 0.0:
            time_cost = [
                f"sample/s: {one_step_samples / one_step_time_cost:.{self._log_prec}f}",
                f"iter/s: {1.0 / one_step_time_cost:.{self._log_prec}f}",
            ] + time_cost
        # use tensorboard log info
        if self._use_tensorboard:
            lr_info = [
                f"lr_total: {self._optimizer.param_groups[0]['lr']}",
                f"lr_per_sample: {self._lr_per_sample}",
            ]
            self._summary.log_scaler(loss_info_list, global_step)
            self._summary.log_scaler(time_cost, global_step)
            self._summary.log_scaler(lr_info, global_step)
            if self._use_log_grad:
                self._summary.log_grad(self._full_model[0], global_step)

        time_cost = [
            f"time_consumed: {time_consumed.__str__()}",
            f"eta_training: {eta_training.__str__()}",
            f"eta_curr_epoch: {eta_curr_epoch.__str__()}",
        ] + time_cost

        # basic info
        basic_info = [
            f"run_name: {self._run_name}",
            f"rank: {self._rank}",
            f"word_size: {self._gpu_num}",
            f"batch_per_gpu: {self._bs_per_gpu}",
            f"lr_total: {self._optimizer.param_groups[0]['lr']}",
            f"lr_per_sample: {self._lr_per_sample}",
            f"sample_per_epoch: {self._sample_per_epoch}",
            f"sample_num_per_gpu: {self._sample_num_per_gpu}",
            f"epoch: {self._epoch_idx}/{self._total_epochs}, "
            f"{int(self._epoch_idx / self._total_epochs * 100)}%",
            f"step: {global_step}/{self._total_steps}, "
            f"{int(global_step / self._total_steps * 100)}%",
            f"total_norm: {self._total_norm}",
        ]

        for data_row_list in itertools.zip_longest(
            basic_info, loss_info_list, time_cost, fillvalue=""
        ):
            table.add_row(data_row_list)
        return f"\n{table.get_string()}"

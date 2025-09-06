# -*- coding: utf-8 -*-
"""Default Config File of Experiments."""
import torch


# deterministic seed
seed = 666
BATCH_SIZE = 20
# define trainer params
trainer = dict(
    type="ParallelTrainer",
    total_epochs=90,
    print_freq_by_step=20,
    save_ckpt_freq_by_epoch=89,
    save_ckpt_freq_by_step=None,
    max_save_ckpt_num=3,
    use_fp16=True,
    use_model_channel_last=True,
    use_sync_bn=False,
    use_tensorboard=True,
    ddp_params=dict(
        bucket_cap_mb=25,
        gradient_as_bucket_view=False,
        find_unused_parameters=False,
        static_graph=True,
    ),
    ddp_type="local",
    data_parallel_random_init=False,
    accumulate_allreduce_grads_in_fp32=True,
    use_contiguous_buffers_in_local_ddp=True,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    megatron_optimizer_params=None,
    num_micro_batch=1,
    clip_gradient=35,
    lr_per_sample=0.001 / 12,  # 0.256/256
    seed=seed,
    use_data_stream=True,
    use_eval_mode=False,
    eval_ckpt_list=[
    ],
)

# define lr scheduler params
lr_scheduler = dict(
    type="CosineLrScheduler",
    total_epoch=90,
    min_lr=0,
)

# define optimizer params
optimizer = dict(
    type="SGD",
    momentum=0.875,
    weight_decay=3.0517578125e-05,
    enable_bn_weight_decay=True,
    nesterov=False,
)
# define p2p_com
p2p_com = dict(
    params_dtype=torch.float16,
    fp32_residual_connection=False,
    use_ring_exchange_p2p=False,
)


# define model params
model_flow = dict(
    type="ModelFlow",
    full_model_cfg=dict(
        type="ParallelRes50V3",
        sub_module_cfg=dict(
            type="ParallelResNet50V3",
            train_mode=True,
            use_sync_bn=False,
        ),
        batch_size=BATCH_SIZE,
    ),
    loss_func_cfg=dict(
        type="LabelSmoothLoss",
        smooth_value=0.1,
    ),
    train_mode=True,
)

# define optimizer params
dataloader = dict(
    type="URDataLoader",
    dataset_cfg=dict(
        type="ImageNetDataset",
        mode="train",
        transforms=[
            dict(
                type="MultiScaleCrop",
                output_size=224,
                scales=[1, 0.875, 0.75, 0.66],
            ),
            dict(
                type="RandomHorizontalFlip",
            ),
            dict(
                type="ToTorchTensor",
            ),
            dict(
                type="Normalize",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        meta_file_key="imagenet",
    ),
    sampler_cfg=dict(
        type="URDistributedSampler",
        shuffle=True,
        seed=seed,
        drop_last=True,
        indices=None,
    ),
    batch_size=BATCH_SIZE,
    seed=seed,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    to_cuda=True,
    drop_last=True,
)

# define ckpt params
ckpt = dict(
    pretrain_model=None,
    ckpt2model_json=None,
    to_cuda=False,
)


# ======================== infer =======================

infer_dataloader = dict(
    type="URDataLoader",
    dataset_cfg=dict(
        type="ImageNetDataset",
        mode="val",
        transforms=[
            dict(
                type="Scale",
                scale_size=256,
            ),
            dict(
                type="CenterCrop",
                output_size=224,
            ),
            dict(
                type="ToTorchTensor",
            ),
            dict(
                type="Normalize",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
        meta_file_key="imagenet",
    ),
    sampler_cfg=dict(
        type="URDistributedSampler",
        shuffle=False,
        seed=seed,
        drop_last=False,
        indices=None,
    ),
    batch_size=BATCH_SIZE,
    seed=seed,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    to_cuda=True,
    drop_last=False,
)

evaluator = dict(
    type="ImageNetEvaluator",
    use_gpu=True,
)

infer_ckpt = dict(
    pretrain_model=None,
    ckpt2model_json=None,
    to_cuda=False,
)

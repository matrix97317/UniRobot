# -*- coding: utf-8 -*-
"""Default Config File of Experiments."""
import torch


# ============Robot Config================
robot = dict(
    type="So101",
    mode="teleopreation",
    fps=25,
    sensor_cfg=dict(
        top=dict(
            type="OpenCVCamera",
            host_name="top",
            port="/dev/video0",
            fps=30,
            width=640,
            height=480,
            color_mode="BGR",
            warmup=True,
            rotate="0",
        ),
        hand=dict(
            type="OpenCVCamera",
            host_name="hand",
            port="/dev/video2",
            fps=30,
            width=640,
            height=480,
            color_mode="BGR",
            warmup=True,
            rotate="180",
        ),
    ),
    motor_cfg=dict(
        type="SoArm101Follower",
        host_name="so_arm101_follower",
        port="/dev/ttyACM1",
        use_degrees=False,
        max_relative_target=None,
        disable_torque_on_disconnect=True,
    ),
    teleoperator_cfg=dict(
        type="SoArm101Leader",
        host_name="so_arm101_leader",
        port="/dev/ttyACM0",
        use_degrees=False,
    ),
    model_cfg=None,
)

# ============Brain Config================

# deterministic seed
seed = 666
BATCH_SIZE = 4
EPOCH = 16000
# define trainer params
trainer = dict(
    type="ParallelTrainer",
    total_epochs=EPOCH,
    print_freq_by_step=20,
    save_ckpt_freq_by_epoch=5,
    save_ckpt_freq_by_step=None,
    max_save_ckpt_num=3,
    use_fp16=False,
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
    lr_per_sample=0.001 / 1,  # 0.256/256
    seed=seed,
    use_data_stream=True,
    use_eval_mode=False,
    eval_ckpt_list=[],
)

# define lr scheduler params
lr_scheduler = dict(
    type="CosineLrScheduler",
    total_epoch=EPOCH,
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
HIDDEN_DIM = 512
model_flow = dict(
    type="ModelFlow",
    full_model_cfg=dict(
        type="ACTModel",
        camera_names=["top", "hand"],
        state_dim=6,
        chunk_size=40,
        sub_module_cfg=dict(
            backbone=dict(
                type="ResNet18",
                pretrain_model="/root/autodl-tmp/resnet18-f37072fd.pth",
                num_channels=HIDDEN_DIM,
                use_all_features=True,
                use_pos_encode=True,
                num_pos_feats=HIDDEN_DIM / 2,
                pe_temperature=10000,
                pe_normalize=True,
                pe_scale=None,
            ),
            transformer=dict(
                type="Transformer",
                d_model=HIDDEN_DIM,
                dropout=0.1,
                nhead=8,
                dim_feedforward=3200,
                num_encoder_layers=4,
                num_decoder_layers=7,
                normalize_before=False,
                return_intermediate_dec=True,
            ),
            encoder=dict(
                type="BlockEncoder",
                d_model=HIDDEN_DIM,
                dropout=0.1,
                nheads=8,
                dim_feedforward=2048,
                enc_layers=4,
                pre_norm=False,
            ),
        ),
    ),
    loss_func_cfg=dict(
        type="ACTKLLoss",
        kl_weight=30,
    ),
    train_mode=True,
    unfold_inputs=True,
)

# define optimizer params
dataloader = dict(
    type="URDataLoader",
    dataset_cfg=dict(
        type="ACTDataset",
        mode="train",
        meta_file={
            "so_arm101": {
                "pick_toy": {
                    "num_episodes": 35,
                    "episode_format": "episode_{:d}.hdf5",
                    "train": "/root/autodl-tmp/pick_toy2/",
                    "val": "/root/autodl-tmp/pick_toy2/",
                    "norm_stats": "/root/autodl-tmp/pick_toy2/norm_stats.pkl",
                }
            }
        },
        task_name="pick_toy",
        robot_name="so_arm101",
        camera_names=["top", "hand"],
        transforms=[
            dict(
                type="ToTorchTensor",
            ),
            dict(
                type="Normalize",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
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
    num_workers=4,
    pin_memory=True,
    prefetch_factor=None,
    persistent_workers=False,
    to_cuda=True,
    drop_last=True,
)

# define ckpt params
ckpt = dict(
    pretrain_model=None,
    ckpt2model_json=None,
    to_cuda=True,
)


# ======================== infer =======================
infer = dict(
    type="BaseInfer",
    infer_type="open_loop",
    export_type=None,
    eval_ckpt_list=[
        "/home/None/unirobot_outputs/task_pick_toy/exp_default/baseline4/ckpt/checkpoint_pipeline_rank_0_last.pth.tar"
    ],
    use_kf=False,
    infer_chunk_step=0,
)

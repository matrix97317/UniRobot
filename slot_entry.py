# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""The entry for unirobot."""
unirobot_slot_entry = {
    "console_scripts": [
        "unirobot=unirobot.cli:cli",
    ],
    "brain.dataloader": [
        "URDataLoader=unirobot.brain.data.dataloader.base_dataloader",
    ],
    "brain.dataset": [
        "ImageNetDataset=unirobot.brain.data.dataset.imagenet",
    ],
    "brain.sampler": [
        "URDistributedSampler=unirobot.brain.data.sampler.base_distributed_sampler",
    ],
    "brain.transform": [
        "CenterCrop=unirobot.brain.data.transform.transform2d",
        "Compose=unirobot.brain.data.transform.base_transform",
        "GroupFCSample=unirobot.brain.data.transform.transform2d",
        "MultiScaleCrop=unirobot.brain.data.transform.transform2d",
        "Normalize=unirobot.brain.data.transform.transform2d",
        "RandomHorizontalFlip=unirobot.brain.data.transform.transform2d",
        "Scale=unirobot.brain.data.transform.transform2d",
        "ToTorchTensor=unirobot.brain.data.transform.base_transform",
    ],
    "brain.encoder": [
        "ParallelResNet50V3=unirobot.brain.model.encoder.parallel_resnet_v3",
    ],
    "brain.decoder": [],
    "brain.layer": [],
    "brain.loss": [
        "ComposeWeightedLoss=unirobot.brain.model.loss.compose_weighted_loss",
        "CrossEntropyLoss=unirobot.brain.model.loss.cross_entropy_loss",
        "LabelSmoothLoss=unirobot.brain.model.loss.smooth_loss",
    ],
    "brain.model_flow": [
        "ModelFlow=unirobot.brain.model.base_model_flow",
    ],
    "brain.full_model": [
        "ParallelRes50V3=unirobot.brain.model.full_model.resnet50_parallel_model",
    ],
    "brain.lr_scheduler": [
        "CosineLrScheduler=unirobot.brain.model.optimizer.base_lr_scheduler",
        "EpochLrScheduler=unirobot.brain.model.optimizer.base_lr_scheduler",
    ],
    "brain.optimizer": [],
    "brain.trainer": [
        "ParallelTrainer=unirobot.brain.trainer.parallel_trainer",
    ],
}

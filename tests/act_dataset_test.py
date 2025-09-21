#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for pkg."""
import pytest

from unirobot.utils.unirobot_slot import DATASET
from unirobot.utils.slot_loader import load_brain_slot


@pytest.mark.skip(reason="need act dataset")
def test_act_dataset() -> None:
    """Unit test for act dataset."""
    load_brain_slot()
    dataset_cfg = dict(
        type="ACTDataset",
        mode="train",
        meta_file={
            "so_arm101": {
                "pick_toy": {
                    "num_episodes": 10,
                    "episode_format": "episode_{:d}.hdf5",
                    "train": "/root/workspace/act/datasets/",
                    "val": "/root/workspace/act/datasets/",
                    "norm_stats": "/root/workspace/act/datasets/norm_stats.pkl",
                }
            }
        },
        task_name="pick_toy",
        robot_name="so_arm101",
        camera_names=["top"],
    )
    dataset = DATASET.build(dataset_cfg)
    data = dataset.__getitem__(1)
    assert "actions" in data

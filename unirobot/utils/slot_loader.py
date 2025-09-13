# -*- coding: utf-8 -*-
"""Slot Loader."""
import logging
import sys
from importlib import import_module

from unirobot.utils.unirobot_slot import FULL_MODEL
from unirobot.utils.unirobot_slot import DATALOADER
from unirobot.utils.unirobot_slot import DATASET
from unirobot.utils.unirobot_slot import EVALUATOR
from unirobot.utils.unirobot_slot import DECODER
from unirobot.utils.unirobot_slot import ENCODER
from unirobot.utils.unirobot_slot import INFERRER
from unirobot.utils.unirobot_slot import LAYER
from unirobot.utils.unirobot_slot import LOSS
from unirobot.utils.unirobot_slot import LR_SCHEDULER
from unirobot.utils.unirobot_slot import OPTIMIZER
from unirobot.utils.unirobot_slot import SAMPLER
from unirobot.utils.unirobot_slot import TRAINER
from unirobot.utils.unirobot_slot import TRANSFORM
from unirobot.utils.unirobot_slot import MODEL_FLOW
from unirobot.utils.unirobot_slot import ROBOT
from unirobot.utils.unirobot_slot import TELEOPERATOR
from unirobot.utils.unirobot_slot import SENSOR
from unirobot.utils.unirobot_slot import MOTOR


if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


logger = logging.getLogger(__name__)


BRAIN_SLOT_MAPPING = {
    DATALOADER.name: DATALOADER,
    DATASET.name: DATASET,
    SAMPLER.name: SAMPLER,
    TRANSFORM.name: TRANSFORM,
    EVALUATOR.name: EVALUATOR,
    FULL_MODEL.name: FULL_MODEL,
    DECODER.name: DECODER,
    ENCODER.name: ENCODER,
    LAYER.name: LAYER,
    LOSS.name: LOSS,
    LR_SCHEDULER.name: LR_SCHEDULER,
    OPTIMIZER.name: OPTIMIZER,
    INFERRER.name: INFERRER,
    TRAINER.name: TRAINER,
    MODEL_FLOW.name: MODEL_FLOW,
}

ROBOT_SLOT_MAPPING = {
    ROBOT.name: ROBOT,
    TELEOPERATOR.name: TELEOPERATOR,
    SENSOR.name: SENSOR,
    MOTOR.name: MOTOR,
}


def load_brain_slot() -> None:
    """Load Brain Slot."""
    for slot_key, slot in BRAIN_SLOT_MAPPING.items():
        for entry_point in entry_points(group=slot_key):
            logger.debug(
                "Slot `%s` with name `%s` to `%s` Slot.",
                entry_point.value,
                entry_point.name,
                slot_key,
            )
            module = import_module(entry_point.value)
            slot.push(module=getattr(module, entry_point.name))


def load_robot_slot() -> None:
    """Load Robot Slot."""
    for slot_key, slot in ROBOT_SLOT_MAPPING.items():
        for entry_point in entry_points(group=slot_key):
            logger.debug(
                "Slot `%s` with name `%s` to `%s` Slot.",
                entry_point.value,
                entry_point.name,
                slot_key,
            )
            module = import_module(entry_point.value)
            slot.push(module=getattr(module, entry_point.name))

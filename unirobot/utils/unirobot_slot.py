# -*- coding: utf-8 -*-
"""UniRobot Slot."""
from unirobot.utils.slot_util import Slot


# NOTE: Ordered by logic rather than alphabetically here.

# UniRobot Brain Slot
# Runner
INFERRER = Slot("brain.inferrer")
TRAINER = Slot("brain.trainer")

# Data
DATASET = Slot("brain.dataset")
DATALOADER = Slot("brain.dataloader")
TRANSFORM = Slot("brain.transform")
SAMPLER = Slot("brain.sampler")

# Model
FULL_MODEL = Slot("brain.full_model")
ENCODER = Slot("brain.encoder")
LAYER = Slot("brain.layer")
DECODER = Slot("brain.decoder")
LOSS = Slot("brain.loss")

# Optimizer
LR_SCHEDULER = Slot("brain.lr_scheduler")
OPTIMIZER = Slot("brain.optimizer")

# Evaluator
EVALUATOR = Slot("brain.evaluator")

# model flow
MODEL_FLOW = Slot("brain.model_flow")

# UniRobot Robot Slot
ROBOT = Slot("robot.robot")
TELEOPERATOR = Slot("robot.teleoperator")
SENSOR = Slot("robot.sensor")
MOTOR = Slot("robot.motor")

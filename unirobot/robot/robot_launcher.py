# -*- coding: utf-8 -*-
"""Run for Robot DDP Launcher."""

import datetime
import logging
import random
import sys
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import torch

from unirobot.utils.cfg_parser import PyConfig
from unirobot.utils.exceptions import exception_hook
from unirobot.utils.file_util import FileUtil
from unirobot.utils.log_util import log_file_init
from unirobot.utils.dist_util import get_global_rank
from unirobot.utils.slot_loader import load_robot_slot
from unirobot.utils.unirobot_slot import ROBOT


logger = logging.getLogger(__name__)

DATETIME_FORMAT = "%Y%m%dT%H%M%S"


def run(
    config: str,
    task_name: str,
    run_type: str,
    use_rl: bool,
) -> None:
    """Entry for unirobot."""
    # Set except hook when run ddp_launcher.
    sys.excepthook = exception_hook
    # NOTE: MPI mode gets real global rank, while spawn mode gets fake
    # global_rank=0, just for initialize FileManager and logger.
    global_rank = get_global_rank()
    FileUtil.init(config, task_name, False, global_rank)
    log_file_init("unirobot", global_rank)
    load_robot_slot()
    # parser cfg
    # config_cfg = PyConfig.fromfile(config)
    # print("=====Demo====")

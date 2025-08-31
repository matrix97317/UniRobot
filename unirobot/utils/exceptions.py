# -*- coding: utf-8 -*-
"""UniRobot custom exceptions."""
import logging
import sys
import traceback
from types import TracebackType
from typing import Optional
from typing import Type

from unirobot.utils.dist_util import dist_abort


logger = logging.getLogger(__name__)


def exception_hook(
    ex_type: Type[BaseException],
    ex_value: BaseException,
    ex_trackback: Optional[TracebackType],
) -> None:
    """Global exception hook for logging."""
    log_list = traceback.format_tb(ex_trackback)
    log_list.append(f"{ex_type.__name__}: {ex_value}")
    logger.error("".join(log_list))
    if not ex_trackback:
        raise ex_value
    sys.__excepthook__(ex_type, ex_value, ex_trackback)
    dist_abort()


class UniRobotBaseException(Exception):
    """Base exception for UniRobot."""


class UniRobotDuplicateRunNameException(UniRobotBaseException):
    """Exception for duplicate run name."""


class UniRobotExistingRunDirNotFoundException(UniRobotBaseException):
    """Exception for existing run name not found."""


class UniRobotExpectedCfgFileNotFoundException(UniRobotBaseException):
    """Exception for expected config file not found."""


class UniRobotInconsistentCfgFileException(UniRobotBaseException):
    """Exception for inconsistent config file."""


class UniRobotInvalidCfgNameException(UniRobotBaseException):
    """Exception for invalid cfg_name."""


class UniRobotInvalidCfgPathException(UniRobotBaseException):
    """Exception for invalid cfg_path."""


class UniRobotInvalidExpNameException(UniRobotBaseException):
    """Exception for invalid exp_name."""


class UniRobotInvalidRunNameException(UniRobotBaseException):
    """Exception for invalid run_name."""


class UniRobotInvalidTaskNameException(UniRobotBaseException):
    """Exception for invalid task_name."""


class UniRobotNotInitializedException(UniRobotBaseException):
    """Exception for file manager not initialized."""


class UniRobotOutputDirIsNotLinkException(UniRobotBaseException):
    """Exception for output dir is not link."""

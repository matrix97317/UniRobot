# -*- coding: utf-8 -*-
"""Log Util.

Config loggers with necessary handlers and formatters with support of distributed
systems and experiment management.
"""
import logging
import os
import sys
from typing import Any
from typing import List
from typing import Optional
from typing import TextIO

from coloredlogs import BasicFormatter
from coloredlogs import ColoredFormatter
from prettytable import PrettyTable

from unirobot.utils.dist_util import get_global_rank
from unirobot.utils.file_util import FileUtil
from unirobot.utils.settings import settings


logger = logging.getLogger(__name__)

LOGGING_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def log_init(name: str, *, stream: TextIO = sys.stdout) -> None:
    """Init logger with stream handler for specific name."""
    logger.warning("Initializing logger `%s` with StreamHandler ...", name)

    init_logger = logging.getLogger(name)
    init_logger.setLevel(settings.logging_level)
    init_logger.propagate = False

    rank = get_global_rank()
    if rank != 0:
        logger.info("Exiting logger `%s` initialization for rank `%s` ...", name, rank)
        return

    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(ColoredFormatter(LOGGING_FORMAT))
    init_logger.addHandler(stream_handler)

    logger.info("Logger `%s` initialized with StreamHandler successfully.", name)


def log_file_init(name: str, rank: int = 0) -> None:
    """Init logger with file handler for specific name."""
    logger.info("Initializing logger `%s` with FileHandler ...", name)

    log_dir = FileUtil.get_log_dir()
    log_file_name = f"{rank}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    init_logger = logging.getLogger(name)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(BasicFormatter(LOGGING_FORMAT))
    init_logger.addHandler(file_handler)

    logger.info(
        "Logger `%s` initialized with FileHandler at `%s` successfully.",
        name,
        log_file_path,
    )


def log_as_table(
    data_list: List[Any],
    num_col: int = 5,
    local_logger: Optional[logging.Logger] = None,
) -> None:
    """Log a data list with specific column number."""
    data_lists = [data_list[i : i + num_col] for i in range(0, len(data_list), num_col)]
    if len(data_lists) > 0:
        data_lists[-1].extend([""] * (num_col - len(data_lists[-1])))

    table = PrettyTable()
    table.add_rows(data_lists)

    local_logger = local_logger or logger
    local_logger.info("\n%s", table.get_string(header=False))

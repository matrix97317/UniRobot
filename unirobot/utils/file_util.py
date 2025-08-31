# -*- coding: utf-8 -*-
"""File Util.

Manage interaction with file system, such as reading and writing files with support of
distributed systems.
"""
from __future__ import annotations

import filecmp
import logging
import os
import shutil
from functools import lru_cache
from shutil import ignore_patterns
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Type

from unirobot.utils.constants import LaunchMode
from unirobot.utils.dist_util import dist_barrier
from unirobot.utils.drop_buffer_cache import drop_buffer_cache
from unirobot.utils.exceptions import UniRobotDuplicateRunNameException
from unirobot.utils.exceptions import UniRobotExistingRunDirNotFoundException
from unirobot.utils.exceptions import UniRobotExpectedCfgFileNotFoundException
from unirobot.utils.exceptions import UniRobotInconsistentCfgFileException
from unirobot.utils.exceptions import UniRobotInvalidCfgNameException
from unirobot.utils.exceptions import UniRobotInvalidCfgPathException
from unirobot.utils.exceptions import UniRobotInvalidExpNameException
from unirobot.utils.exceptions import UniRobotInvalidRunNameException
from unirobot.utils.exceptions import UniRobotInvalidTaskNameException
from unirobot.utils.exceptions import UniRobotNotInitializedException
from unirobot.utils.exceptions import UniRobotOutputDirIsNotLinkException
from unirobot.utils.settings import global_settings
from unirobot.utils.settings import settings


logger = logging.getLogger(__name__)

CKPT_FOLDER_NAME = "ckpt"
EXPORT_FOLDER_NAME = "export"
LOG_FOLDER_NAME = "log"
OUTPUT_FOLDER_NAME = "output"
PREFIX_CFG = "cfg_"
PREFIX_EXP = "exp_"
PREFIX_TASK = "task_"
SUMMARY_FOLDER_NAME = "summary"

IGNORE_PATTERNS_FUNC = ignore_patterns(
    ".*",
    "_*",
    OUTPUT_FOLDER_NAME,
)


def need_init(func: Callable[..., Any]) -> Callable[..., Any]:
    """Check initialization."""

    def wrapper(cls: Type[FileUtil], *args: Any, **kwargs: Any) -> Any:
        """Check initialization."""
        if not cls.initialized:
            raise UniRobotNotInitializedException(f"{cls.__name__} is not initialized.")
        return func(cls, *args, **kwargs)

    return wrapper


class FileUtil:
    """File Manager."""

    _cfg_dir: ClassVar[str]
    _cfg_name: ClassVar[str]
    _ckpt_dir: ClassVar[str]
    _exp_name: ClassVar[str]
    _export_dir: ClassVar[str]
    _log_dir: ClassVar[str]
    _run_dir: ClassVar[str]
    _summary_dir: ClassVar[str]
    _task_dir: ClassVar[str]
    _task_name: ClassVar[str]
    initialized: ClassVar[bool] = False

    @classmethod
    def init(
        cls,
        cfg_path: str,
        run_name: str,
        is_resumed: bool = False,
        rank: int = 0,
    ) -> None:
        """Init for file manager."""
        logger.info("Initializing FileUtil...")

        logger.debug(
            "cfg_path: %s, run_name: %s, is_resumed: %s.",
            cfg_path,
            run_name,
            is_resumed,
        )

        if not run_name:
            raise UniRobotInvalidRunNameException(
                f"Invalid run_name `{run_name}` found."
            )

        if not os.path.isfile(cfg_path):
            raise UniRobotInvalidCfgPathException(
                f"cfg_path `{cfg_path}` is not a file."
            )

        cfg_path = os.path.abspath(cfg_path)
        cls._cfg_dir = os.path.dirname(cfg_path)
        cls._cfg_name = os.path.basename(cfg_path)
        cls._exp_name = os.path.basename(cls._cfg_dir)
        cls._task_name = os.path.basename(os.path.dirname(cls._cfg_dir))

        logger.debug(
            "cfg_dir: %s, cfg_name: %s, task_name: %s, exp_name: %s",
            cls._cfg_dir,
            cls._cfg_name,
            cls._task_name,
            cls._exp_name,
        )

        cls._ckpt_dir = _ckpt_dir(cls._task_name, cls._exp_name, run_name)
        cls._export_dir = _export_dir(cls._task_name, cls._exp_name, run_name)
        cls._log_dir = _log_dir(cls._task_name, cls._exp_name, run_name)
        cls._run_dir = _run_dir(cls._task_name, cls._exp_name, run_name)
        cls._summary_dir = _summary_dir(cls._task_name, cls._exp_name, run_name)
        cls._task_dir = _task_dir(cls._task_name)

        # Drop buffer cache for the cfg dir before setup or validation.
        drop_buffer_cache(cls._cfg_dir)

        if rank != 0:
            logger.info("Skip actual file/folder operation for rank `%s`.", rank)
        else:
            if not cls._task_name.startswith(PREFIX_TASK):
                raise UniRobotInvalidTaskNameException(
                    f"Invalid task name: `{cls._task_name}` which should starts "
                    f"with `{PREFIX_TASK}`."
                )
            if not cls._exp_name.startswith(PREFIX_EXP):
                raise UniRobotInvalidExpNameException(
                    f"Invalid experiment name: `{cls._exp_name}` which should starts "
                    f"with `{PREFIX_EXP}`."
                )
            if not cls._cfg_name.startswith(PREFIX_CFG):
                raise UniRobotInvalidCfgNameException(
                    f"Invalid config name: `{cls._cfg_name}` which should starts with "
                    f"`{PREFIX_CFG}`."
                )
            if not is_resumed:
                cls._setup()
            else:
                cls._validate()
            logger.info("FileUtil initialized successfully.")

        cls.initialized = True
        dist_barrier()

    @classmethod
    def _setup(cls) -> None:
        """Set up files, folders and symlinks."""
        logger.info("Start to set up files, folders and symlinks.")

        if os.path.exists(cls._run_dir):
            if settings.launch_mode == LaunchMode.SPAWN:
                logger.warning("Skip duplicate run name check in SPAWN launch mode.")
                return
            # mpi
            raise UniRobotDuplicateRunNameException(
                f"Existing run name found, please use another one or remove it "
                f"manually: `{cls._run_dir}`."
            )

        logger.info("Creating ckpt dir at %s.", cls._ckpt_dir)
        os.makedirs(cls._ckpt_dir, exist_ok=True)
        logger.info("Creating export dir at %s.", cls._export_dir)
        os.makedirs(cls._export_dir, exist_ok=True)
        logger.info("Creating log dir at %s.", cls._log_dir)
        os.makedirs(cls._log_dir, exist_ok=True)
        logger.info("Creating summary dir at %s.", cls._summary_dir)
        os.makedirs(cls._summary_dir, exist_ok=True)

        logger.info(
            "Copying experiment configs from %s to %s.", cls._cfg_dir, cls._run_dir
        )
        shutil.copytree(
            cls._cfg_dir,
            cls._run_dir,
            symlinks=True,
            ignore=IGNORE_PATTERNS_FUNC,
            dirs_exist_ok=True,
        )

        output_dir = os.path.abspath(OUTPUT_FOLDER_NAME)
        logger.info("Creating link for %s at %s", cls._run_dir, output_dir)
        if os.path.exists(output_dir) or os.path.islink(output_dir):
            if not os.path.islink(output_dir):
                raise UniRobotOutputDirIsNotLinkException(
                    f"Folder `{output_dir}` exists and is not a link."
                )
            logger.warning("Unlink existing link `%s`.", output_dir)
            os.unlink(output_dir)
        os.symlink(cls._task_dir, output_dir)

        logger.info("Finish setting up files, folders and symlinks.")

    @classmethod
    def _validate(cls) -> None:
        """Validate consistency when resume."""
        logger.info("Start to validate consistency when resume.")

        if not os.path.exists(cls._run_dir):
            raise UniRobotExistingRunDirNotFoundException(
                f"No existing run dir found. Expected: `{cls._run_dir}`."
            )

        # Drop buffer cache for the run dir before validation.
        drop_buffer_cache(cls._run_dir)

        for dirpath, _, filenames in os.walk(cls._cfg_dir):
            # NOTE: os.walk() traverses into every folders even when their ancestor
            # folders are ignored. So we split the relative dir with path separator and
            # reuse the shutil.ignore_patterns to judge whether they should be ignored.
            #
            # Case: When `output` is ignored, `output/XXX/YYY` should also be ignored.
            if dirpath != cls._cfg_dir and IGNORE_PATTERNS_FUNC(
                _, os.path.relpath(dirpath, cls._cfg_dir).split(os.path.sep)
            ):
                logger.debug("Skip validate folder `%s`", dirpath)
                continue
            # Reuse the shutil.ignore_patterns to distinguish ignored filenames.
            ignore_filenames = IGNORE_PATTERNS_FUNC(dirpath, filenames)
            for filename in filenames:
                if filename in ignore_filenames:
                    logger.debug("Skip validate file `%s` in `%s`", filename, dirpath)
                    continue
                file_path = os.path.abspath(os.path.join(dirpath, filename))
                relative_path = os.path.relpath(file_path, cls._cfg_dir)
                expected_path = os.path.join(cls._run_dir, relative_path)
                if not os.path.exists(expected_path):
                    raise UniRobotExpectedCfgFileNotFoundException(
                        f"Expected config file not found when resume: `{file_path}`. "
                        f"Expected: `{expected_path}`."
                    )
                if not filecmp.cmp(file_path, expected_path):
                    raise UniRobotInconsistentCfgFileException(
                        f"Inconsistent config file found when resume: `{file_path}` "
                        f"vs. `{expected_path}`."
                    )

        logger.info("Finish validating consistency when resume.")

    @classmethod
    @need_init
    def get_run_dir(cls) -> str:
        """Get the run dir."""
        return cls._run_dir

    @classmethod
    @need_init
    def get_ckpt_dir(cls) -> str:
        """Get the ckpt dir."""
        return cls._ckpt_dir

    @classmethod
    @need_init
    def get_export_dir(cls) -> str:
        """Get the export dir."""
        return cls._export_dir

    @classmethod
    @need_init
    def get_log_dir(cls) -> str:
        """Get the log dir."""
        return cls._log_dir

    @classmethod
    @need_init
    def get_summary_dir(cls) -> str:
        """Get the summary dir."""
        return cls._summary_dir


@lru_cache
def _task_dir(task_name: str) -> str:
    """Return output dir for task with cache."""
    # NOTE: To be compatible with `as_root` mode on training platform.
    output_dir = settings.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(
            "/home",
            str(global_settings.gitlab_user_login or global_settings.user),
            settings.output_dir,
        )
    return os.path.join(output_dir, task_name)


@lru_cache
def _exp_dir(task_name: str, exp_name: str) -> str:
    """Return output dir for experiment with cache."""
    return os.path.join(_task_dir(task_name), exp_name)


@lru_cache
def _run_dir(task_name: str, exp_name: str, run_name: str) -> str:
    """Return output dir for run with cache."""
    return os.path.join(_exp_dir(task_name, exp_name), run_name)


@lru_cache
def _ckpt_dir(task_name: str, exp_name: str, run_name: str) -> str:
    """Return ckpt dir with cache."""
    return os.path.join(_run_dir(task_name, exp_name, run_name), CKPT_FOLDER_NAME)


@lru_cache
def _export_dir(task_name: str, exp_name: str, run_name: str) -> str:
    """Return export dir with cache."""
    return os.path.join(_run_dir(task_name, exp_name, run_name), EXPORT_FOLDER_NAME)


@lru_cache
def _log_dir(task_name: str, exp_name: str, run_name: str) -> str:
    """Return log dir with cache."""
    return os.path.join(_run_dir(task_name, exp_name, run_name), LOG_FOLDER_NAME)


@lru_cache
def _summary_dir(task_name: str, exp_name: str, run_name: str) -> str:
    """Return summary dir with cache."""
    return os.path.join(_run_dir(task_name, exp_name, run_name), SUMMARY_FOLDER_NAME)

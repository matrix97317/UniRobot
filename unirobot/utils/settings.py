# -*- coding: utf-8 -*-
"""Settings for unirobot.

Manage environment variable based settings with default values.
"""
import logging
from logging import getLevelName
from typing import Optional

from pydantic_settings import BaseSettings

from unirobot.utils.constants import Backend
from unirobot.utils.constants import LaunchMode


class Settings(BaseSettings):  # pylint: disable=too-few-public-methods
    """Settings of unirobot."""

    backend: Optional[Backend] = None
    data_path_prefix: str = ""
    launch_mode: LaunchMode = LaunchMode.MPI
    logging_level: str = getLevelName(logging.INFO)
    output_dir: str = "unirobot_outputs"
    com_timeout: int = 1800

    class Config:  # pylint: disable=too-few-public-methods
        """Config for settings."""

        env_prefix = "UNIROBOT_"


class GlobalSettings(BaseSettings):  # pylint: disable=too-few-public-methods
    """Settings for the global environment."""

    ci: bool = False
    cuda_visible_devices: Optional[str] = None
    gitlab_user_login: Optional[str] = None
    user: Optional[str] = None
    vc_master_hosts: str = "localhost"


settings = Settings()
global_settings = GlobalSettings()

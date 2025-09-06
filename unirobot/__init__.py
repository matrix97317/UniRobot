# -*- coding: utf-8 -*-
"""unirobot."""
import logging
from importlib.metadata import version

from unirobot.utils.dist_util import disable_kmp_affinity
from unirobot.utils.dist_util import init_set_device
from unirobot.utils.log_util import log_init


__version__ = version(__package__)


log_init(__name__)
logger = logging.getLogger(__name__)


init_set_device()
disable_kmp_affinity()

# -*- coding: utf-8 -*-
"""Constants for UniRobot."""

from enum import Enum


seed: int = 666
run_name: str = "E0001"


class Backend(Enum):
    """Backend Enum."""

    GLOO = "gloo"
    MPI = "mpi"
    NCCL = "nccl"


class LaunchMode(Enum):
    """Launch Mode Enum."""

    MPI = "mpi"
    SPAWN = "spawn"

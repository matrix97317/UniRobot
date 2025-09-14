# -*- coding: utf-8 -*-
"""Robot's Device interface."""
import logging
from abc import ABC
from typing import Any


logger = logging.getLogger(__name__)


class BaseDevice(ABC):
    """The abstract interface of Robot Device.

    Args:
        host_name (str): Device's ID, such as IP.
        port (str) : Device' Port, such as IP's port, UART prot.
    """

    def __init__(self, host_name: str = "localhost", port: str = "1234", **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self._host_name = host_name
        self._port = port

    def __str__(self) -> str:
        """Return str."""
        return f"{self._host_name}({self._port})"

    def open(self, *args, **kwargs) -> Any:
        """Open a device."""
        raise NotImplementedError()

    def configure(self, *args, **kwargs) -> Any:
        """Configure a device."""
        raise NotImplementedError()

    def get(self, *args, **kwargs) -> Any:
        """Get info from device."""
        raise NotImplementedError()

    def put(self, *args, **kwargs) -> Any:
        """Put info to device."""
        raise NotImplementedError()

    def close(self, *args, **kwargs) -> Any:
        """Close a device."""
        raise NotImplementedError()

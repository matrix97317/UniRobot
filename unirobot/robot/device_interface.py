# -*- coding: utf-8 -*-
"""Robot's Device interface."""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


logger = logging.getLogger(__name__)


class BaseDevice(ABC):
    """The abstract interface of Robot Device.

    Args:
        host_name (str): Device's ID, such as IP.
        port (str) : Device' Port, such as IP's port, UART prot.
    """

    def __init__(self, host_name: str = "localhost", port: str = "1234"):
        """Init."""
        self._host_name = host_name
        self._port = port

    def open(self, *args, **kwargs) -> None:
        """Open a device."""
        raise NotImplementedError()

    def configure(self, *args, **kwargs) -> None:
        """Configure a device."""
        raise NotImplementedError()

    def get(self, *args, **kwargs) -> None:
        """Get info from device."""
        raise NotImplementedError()

    def put(self, *args, **kwargs) -> None:
        """Put info to device."""
        raise NotImplementedError()

    def close(self, *args, **kwargs) -> None:
        """Close a device."""
        raise NotImplementedError()

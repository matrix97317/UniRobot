# -*- coding: utf-8 -*-
"""Summary: cfg_parser.py.

cfg_parser.py is used to parse \
the experimental configuration parameter file (exp_xxx.py) \
into a dictionary sequence, which supports dictionary access and `.` access.
"""

import os.path as osp
import sys
from importlib import import_module

from addict import Dict


class ConfigDict(Dict):
    """A config container, so that it can be used in a dict way.

    Modify from mmcv.
    """

    def __missing__(self, name):
        """Raise miss key error."""
        raise KeyError(name)

    def __getattr__(self, name):
        """Get dict attribute value by name."""
        try:
            value = super().__getattr__(name)
            return value
        except KeyError as ex:
            raise AttributeError(
                f"{self.__class__.__name__ } object has no attribute {name}",
            ) from ex


class PyConfig:
    """A facility for config and config files.

    Modify from mmcv.

    The interface is the same as a dict object and also allows access config
    values as attributes.

    """

    @staticmethod
    def fromfile(filename):
        """Parse Python file as DICT."""
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise KeyError(f"file {filename} does not exist")
        if filename.endswith(".py"):
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)

            old_module = None
            if module_name in sys.modules:
                old_module = sys.modules.pop(module_name)

            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }
            # IMPORTANT: pop to avoid `import_module` from cache, to avoid the
            # cfg sharing by multiple processes or functions, which may cause
            # interference and get unexpected result.
            sys.modules.pop(module_name)

            if old_module is not None:
                sys.modules[module_name] = old_module
        else:
            raise IOError("Only py type are supported now!")
        return PyConfig(cfg_dict, filename=filename)

    def __init__(self, cfg_dict=None, filename=None, encoding="utf-8"):
        """Init PyConfig."""
        if cfg_dict is None:
            cfg_dict = {}
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f"cfg_dict must be a dict, but got {type(cfg_dict)}")

        super().__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super().__setattr__("_filename", filename)
        if filename:
            with open(filename, "r", encoding=encoding) as fin:
                super().__setattr__("_text", fin.read())
        else:
            super().__setattr__("_text", "")

    @property
    def filename(self):
        """Return Python file name."""
        return self._filename

    @property
    def text(self):
        """Return the text form of the parameter file."""
        return self._text

    def __repr__(self):
        """Show PyConfig Info."""
        return f"PyConfig (path: { self.filename}):{self._cfg_dict.__repr__()}"

    def __len__(self):
        """Return the length of paramerter dict."""
        return len(self._cfg_dict)

    def __getattr__(self, name):
        """Get value By attribute name."""
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        """Get value By name."""
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        """Setvalue of attribute."""
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        """Set value of name."""
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        """Return iterable object."""
        return iter(self._cfg_dict)

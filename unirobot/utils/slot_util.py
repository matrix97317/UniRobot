# -*- coding: utf-8 -*-
"""Summary: slot.py.

slot.py is used to \
instantiate classes during system initialization, \
with the help of decorators.
"""

import inspect
import logging
from functools import partial
from importlib import import_module
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


logger = logging.getLogger(__name__)


class Slot:
    """A slot to map strings to classes.

    Registered(Pushed) object could be built from slot.

    Args:
        name (str): Slot name.
        build_func(func, optional): Build function to construct instance from
            Slot, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Slot, optional): Parent slot. The class registered in
            children slot could be built from parent. Default: None.
        scope (str, optional): The scope of slot. It is the key to search
            for children slot. If not specified, scope will be the name of
            the package where class is defined.
            Default: None.

    Example:
        >>> MODELS = Slot('models')
        >>> @MODELS.push()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))
    """

    def __init__(
        self,
        name: str,
        build_func: Optional[Callable] = None,
        parent: Any = None,
        scope: Optional[str] = None,
    ) -> None:
        """Init Slot Class."""
        self._name = name
        self._module_dict: Dict = {}
        self._children: Dict = {}
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func: Callable = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            if not isinstance(parent, Slot):
                raise RuntimeError(f"parent type {type(parent)} not is Slot.")
            parent._add_children(self)
            self.parent: Union[Slot, None] = parent
        else:
            self.parent = None

    def __len__(self) -> int:
        """Return the number of all registed module."""
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        """Check whether it exists according to the key."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        """Show all registed module info."""
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @staticmethod
    def infer_scope() -> str:
        """Infer the scope of slot.

        The name of the package where slot is defined will be returned.
        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key: str) -> Tuple[Union[str, None], str]:
        """Split scope and key.

        The first scope will be split from key.

        Returns:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        return None, key

    @property
    def name(self) -> str:
        """Return slot name."""
        return self._name

    @property
    def scope(self) -> str:
        """Return slot scope."""
        return self._scope

    @property
    def module_dict(self) -> Dict:
        """Return slot modules."""
        return self._module_dict

    @property
    def children(self) -> Dict:
        """Return child module form current module."""
        return self._children

    def get(self, key: str) -> Any:
        """Get the slot record.

        Args:
            key (str): The class name in string format.
        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
            raise ValueError(f"{real_key} not in Register {self._name}.")

        # get from self._children
        if scope in self._children:
            return self._children[scope].get(real_key)
        # goto root
        parent = self.parent
        while parent.parent is not None:  # type: ignore[union-attr]
            parent = parent.parent  # type: ignore[union-attr]
        return parent.get(key)  # type: ignore[union-attr]

    def build(self, *args, **kwargs):
        """Instantiate class form pushed moudle list."""
        return self.build_func(*args, **kwargs, slot=self)

    def _add_children(self, slot: Any) -> None:
        """Add children for a slot.

        The ``slot`` will be added as children based on its scope.
        The parent slot could build objects from children slot.
        """
        if not isinstance(slot, Slot):
            raise TypeError(f"parent slot type {slot} is not Slot.")
        if slot.scope is None:
            raise ValueError("slot.scope is None.")

        if slot.scope in self.children:
            raise RuntimeError(f"scope {slot.scope} exists in {self.name} slot")
        self.children[slot.scope] = slot

    def _push(
        self,
        module_class,
        module_name: Optional[Union[List[str], str]] = None,
        force: bool = False,
    ) -> None:
        """Add module into slot."""
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            module_class_tuple = (module_class.__name__, module_class.__module__)
            if (
                not force
                and name in self._module_dict
                and module_class_tuple != self._module_dict[name]
            ):
                raise KeyError(f"{name} is already registered in {self.name}.")
            self._module_dict[name] = module_class_tuple

    def deprecated_push(self, cls=None, force: bool = False) -> Any:
        """Old slot module."""
        logger.error(
            "The old API of push(module, force=False) \
            is deprecated and will be removed, please use the new API \
            push(name=None, force=False, module=None) instead.",
        )
        if cls is None:
            return partial(self.deprecated_push, force=force)
        self._push(cls, force=force)
        return cls

    def push(
        self,
        name: Optional[str] = None,
        force: bool = False,
        module=None,
    ) -> Any:
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.

        Example:
            >>> backbones = Slot('backbone')
            >>> @backbones.push()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Slot('backbone')
            >>> @backbones.push(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Slot('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.push(ResNet)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_push(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence",
                f"  of str, but got {type(name)}",
            )

        # use it as a normal method: x.push(module=SomeClass)
        if module is not None:
            self._push(module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.push()
        def _register(cls):
            self._push(module_class=cls, module_name=name, force=force)
            return cls

        return _register


def build_from_cfg(
    cfg: Dict,
    slot: Slot,
    default_args: Optional[Dict] = None,
) -> Any:
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        slot (:obj:`Slot`): The slot to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f"but got {cfg}\n{default_args}",
            )
    if not isinstance(slot, Slot):
        raise TypeError(
            f"slot must be an Slot object, but got {type(slot)}",
        )
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            f"default_args must be a dict or None, but got {type(default_args)}",
        )

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type: Any = args.pop("type")
    if isinstance(obj_type, str):
        # pylint: disable=fixme
        obj_cls = slot.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {slot.name} slot")
        module_name, module_path = obj_cls
        module_obj = import_module(module_path)
        obj_cls = getattr(module_obj, module_name)
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    return obj_cls(**args)

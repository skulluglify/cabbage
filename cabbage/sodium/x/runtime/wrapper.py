#!/usr/bin/env python3
import json
from abc import ABC, abstractmethod
from types import TracebackType
from typing import TypeVar, Type, Dict, Any, ParamSpec, Protocol

BaseClassSelf = TypeVar("BaseClassSelf", bound="BaseClass")
BaseClassType = Type[BaseClassSelf]


_P = ParamSpec("_P")
_VT = TypeVar("_VT")
WrapperSelf = TypeVar("WrapperSelf", bound="Wrapper")
WrapperType = Type[WrapperSelf]


class _WrapperFunc(Protocol):
    def __call__(self: WrapperSelf, *args: _P.args, **kwargs: _P.kwargs) -> _VT: ...


class Wrapper:
    """
        Make Wrapper from Any Instances Or Function.
    """

    _func_wrapper: _WrapperFunc

    def __init__(self, obj: _WrapperFunc | _VT):
        """
            Make Wrapper from Any Instances Or Function.
        :param obj:
        """

        if not isinstance(obj, type):
            obj = obj.__class__

        if not callable(obj):
            raise Exception(f"Type '{obj}' is not callable")

        self._func_wrapper = obj

    def __call__(self: WrapperSelf, *args: _P.args, **kwargs: _P.kwargs) -> _VT:
        return self._func_wrapper(*args, **kwargs)

    # def __get__(self, instance, owner): ...
    # def __set__(self, instance, value): ...
    # def __del__(self): ...


class BaseClass(ABC):
    """
        BaseWrapper, Basic Wrapper.
        Passing Default Equipment.
        property like name, enter, and exit.
    """
    exit_code = 0

    @classmethod
    def create(cls: BaseClassType, *args: _P.args, **kwargs: _P.kwargs) -> BaseClassSelf:
        """
            Create Base Wrapper.
        :return:
        """

        return Wrapper(cls)(*args, **kwargs)

    @property
    def name(self: BaseClassSelf):
        """
            Get Name of Base Class.
        :return:
        """

        # Collection Main Class and Base Class
        cls = self.__class__
        base = cls.__base__

        # return f"{cls.__qualname__}({base.__name__})"
        return f"{cls.__name__}({base.__name__})"

    def __enter__(self: BaseClassSelf) -> BaseClassSelf:
        return self

    def __exit__(self: BaseClassSelf,
                 exc_type: Type[BaseException],
                 exc_val: BaseException, exc_tb: TracebackType) -> None:
        return None

    def close(self: BaseClassSelf) -> int:
        """
            Closing with EXIT_SUCCESS.
        :return:
        """
        return self.exit_code


BaseConfigSelf = TypeVar("BaseConfigSelf", bound="BaseConfig")
BaseConfigType = Type[BaseConfigSelf]


class BaseConfig(ABC):
    """
        Base Configuration for Any Data.
        Suitable Data Parsing for JSON format.
    """

    @abstractmethod
    def merge(self: BaseConfigSelf, config: BaseConfigSelf) -> BaseConfigSelf: ...

    @abstractmethod
    def copy(self: BaseConfigSelf) -> BaseConfigSelf: ...

    @classmethod
    @abstractmethod
    def from_dict(cls: BaseConfigType, data: Dict[str, Any], *args: _P.args, **kwargs: _P.kwargs) -> BaseConfigType: ...

    @abstractmethod
    def to_dict(self: BaseConfigSelf, *args: _P.args, **kwargs: _P.kwargs) -> Dict[str, Any]: ...

    @classmethod
    def from_json(cls: BaseConfigType, data: str | bytes, *args: _P.args, **kwargs: _P.kwargs) -> BaseConfigSelf:
        return cls.from_dict(json.loads(data), *args, **kwargs)

    def to_json(self: BaseConfigSelf, *args: _P.args, **kwargs: _P.kwargs) -> str:
        return json.dumps(self.to_dict(), *args, **kwargs)

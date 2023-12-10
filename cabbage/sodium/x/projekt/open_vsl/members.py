#!/usr/bin/env python3
from abc import abstractmethod, ABC
from typing import ParamSpec, TypeVar, Type

vFunctionSelf = TypeVar("vFunctionSelf", bound="vFunction")
vFunctionType = Type[vFunctionSelf]

vBind = TypeVar("vBind")
vReturn = TypeVar("vReturn")
vParams = ParamSpec("vParams")
vArgType = vParams.args
vKwArgType = vParams.kwargs


class vFunction(ABC):
    """
        New Function Proposals

        > func(this, *args, **kwargs)

        or

        > func(self, *args, **kwargs)

        Implemented Function Builder for Create New Object Class
        Without Create Complex Code With Class Method.
    """

    @abstractmethod
    def __call__(self: vFunctionSelf, bind: vBind, *args: vArgType, **kwargs: vKwArgType) -> vReturn: ...

    def apply(self: vFunctionSelf, bind: vBind, *args: vArgType, **kwargs: vKwArgType) -> vReturn:
        return self.__call__(bind, *args, **kwargs)


class vFunctionMember:
    """
        New Function Member

        Suitable for Single Function Common
        Or Make it Combine with Class Method.
    """

    bind: vBind
    call: vFunction
    args: vArgType
    kwargs: vKwArgType

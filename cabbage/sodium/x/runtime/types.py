#!/usr/bin/env python3
from typing import Any, Dict, TypeVar, Type

from sodium.x.runtime.wrapper import BaseClass

VoidSelf = TypeVar("VoidSelf", bound="Void")
VoidType = Type[VoidSelf]


class Void(BaseClass):

    def __init__(self: VoidSelf, *args: Any, **kwargs: Any):
        """
            Void Empty.
        :param args:
        :param kwargs:
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """
            Callable Purposes.
        :param args:
        :param kwargs:
        :return:
        """
        return None

    def __repr__(self) -> str:
        """
            Represent Void Name.
        :return:
        """
        return "<Void>"

    def __str__(self) -> str:
        """
            Void like Empty String.
        :return:
        """
        return ""


void = Void

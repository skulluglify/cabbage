#!/usr/bin/env python3
# TODO: name/names can be empty string

from types import UnionType
from typing import Any, Dict, List, Tuple, Type, TypeVar, Sequence

# Utilities

_VT = TypeVar("_VT")
TypeAny = Type[Any] | UnionType | Tuple[Type[Any] | UnionType | Tuple[Any, ...], ...]
Named = List[str] | Tuple[str, ...] | str


def ref_obj_get_val(obj: object, name: str, default: _VT | None = None) -> _VT | None:
    """
        Reference Object Get Value By Name.
    :param obj:
    :param name:
    :param default:
    :return:
    """

    if isinstance(obj, Dict):
        if name in obj:
            return obj[name]

    if isinstance(obj, Sequence):
        if isinstance(obj, List) or isinstance(obj, Tuple):
            index: int

            try:
                index = int(name)

            except ValueError:
                return default

            # Out Of Bound.
            if len(obj) <= index:
                return default

            return obj[index]

    if hasattr(obj, name):
        return getattr(obj, name, default)

    return default


def refs_obj_get_val(obj: object, names: Named, default: _VT | None = None) -> _VT | None:
    """
        References Object Get Value By Separate Names.
    :param obj:
    :param names:
    :param default:
    :return:
    """

    if isinstance(names, str):
        names = names.split(".")

    val_store = obj
    for name in names:
        val_store = ref_obj_get_val(val_store, name, None)
        if val_store is None:
            return default

    return val_store


def ref_obj_set_val(obj: object, name: str, value: _VT) -> bool:
    """
        References Object Set Value By Name.
    :param obj:
    :param name:
    :param value:
    :return:
    """

    if isinstance(obj, Dict):
        if name in obj:
            obj[name] = value
            return True

    if isinstance(obj, Sequence):
        if isinstance(obj, List) or isinstance(obj, Tuple):
            index: int

            try:
                index = int(name)

            except ValueError:
                return False

            # Out Of Bound.
            if len(obj) <= index:
                return False

            # Tuple Not Support Item Assignment.
            if isinstance(obj, List):
                obj[index] = value
                return True

    if hasattr(obj, name):
        setattr(obj, name, value)
        return True

    return False


def refs_obj_set_val(obj: object, names: Named, value: _VT) -> bool:
    """
        References Object Set Value By Name.
    :param obj:
    :param names:
    :param value:
    :return:
    """

    if isinstance(names, str):
        names = names.split(".")

    n = len(names)

    if 0 < n:
        if 1 < n:
            prefix = names[:n - 1]
            suffix = names[-1]

            obj_ref = refs_obj_get_val(obj, prefix)
            if obj_ref is None:
                return False

            return ref_obj_set_val(obj_ref, suffix, value)

        prefix = names[0]
        return ref_obj_set_val(obj, prefix, value)

    return False


def ref_obj_type_safe(obj: object, name: str, __type: TypeAny, ignore_errors: bool = False) -> bool:
    """
        Reference Object Type Safe.
    :param obj:
    :param name:
    :param __type:
    :param ignore_errors:
    :return:
    """
    val = ref_obj_get_val(obj, name)

    if val is None:
        if not ignore_errors:
            raise Exception("key '{}' is not found".format(name))

        return False

    if not isinstance(val, __type):
        if not ignore_errors:
            raise Exception("key '{}' is not '{}'".format(name, __type))

        return False

    return True


def refs_obj_type_safe(obj: object, names: Named, __type: TypeAny, ignore_errors: bool = False) -> bool:
    """
        References Object Type Safe.
    :param obj:
    :param names:
    :param __type:
    :param ignore_errors:
    :return:
    """

    key_names: str
    if isinstance(names, Sequence):
        key_names = ".".join(names)

    else:
        key_names = names
        names = names.split(".")

    val = refs_obj_get_val(obj, names)

    if val is None:
        if not ignore_errors:
            raise Exception("key '{}' is not found".format(key_names))

        return False

    if not isinstance(val, __type):
        if not ignore_errors:
            raise Exception("key '{}' is not '{}'".format(key_names, __type))

        return False

    return True

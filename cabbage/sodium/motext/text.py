#!/usr/bin/env python3
from .pipe_text import (pipe_text_cew,
                        pipe_text_num_only,
                        pipe_text_to_lower,
                        pipe_text_to_upper,
                        pipe_text_safe_name, pipe_text_safe_name_kv)


def text_safe_name(value: str, extensible: bool = False) -> str:
    """
        Capitalize Each World.
    :param value:
    :param extensible:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_safe_name(map(ord, value), extensible=extensible))))


def text_cew(value: str) -> str:
    """
        Capitalize Each World.
    :param value:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_cew(map(ord, value)))))


def text_to_lower(value: str) -> str:
    """
        Capitalize Each World.
    :param value:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_to_lower(map(ord, value)))))


def text_to_upper(value: str) -> str:
    """
        Capitalize Each World.
    :param value:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_to_upper(map(ord, value)))))


def text_num_only(value: str, floating: bool = False) -> str:
    """
        Capitalize Each World.
    :param value:
    :param floating:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_num_only(map(ord, value), floating=floating))))


def text_safe_name_kv(value: str, snippet: bool = False) -> str:
    """
        Capitalize Each World.
    :param value:
    :param snippet:
    :return:
    """

    return "".join(tuple(map(chr, pipe_text_safe_name_kv(map(ord, value), snippet=snippet))))

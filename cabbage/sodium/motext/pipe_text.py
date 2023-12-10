#!/usr/bin/env python3
from typing import Iterable, Iterator


# text pipelines
def pipe_text_safe_name(codepoints: Iterable[int], extensible: bool = False) -> Iterator[int]:
    """
        Skipping Another Symbol Except ASCII.
    :param codepoints:
    :param extensible:
    :return:
    """

    for codepoint in codepoints:

        if extensible:  # like file.

            # dots.
            if codepoint in (44, 46):  # ",", "."
                yield 46

            # spaces.
            if codepoint in (32, 43, 45, 95):  # " ", "+", "-", "_"
                if codepoint not in (43,):  # except char "+"
                    yield codepoint
                    continue

                yield 32  # prevent char "+" to " "

        else:
            # spaces.
            if codepoint in (32, 43, 45, 95):  # " ", "+", "-", "_"
                yield 32

        # nums.
        if 48 <= codepoint <= 57:
            yield codepoint

        # upper cases.
        if 65 <= codepoint <= 90:
            yield codepoint

        # lower cases.
        if 97 <= codepoint <= 122:
            yield codepoint


def pipe_text_cew(codepoints: Iterable[int]) -> Iterator[int]:
    """
        Capitalize Each World.
    :param codepoints:
    :return:
    """

    capitalized = True
    for codepoint in codepoints:

        # neither alpha upper and lower.
        if not (97 <= codepoint <= 122 and 65 <= codepoint <= 90):
            capitalized = True  # reset capitalized.
            yield codepoint
            continue

        # to upper case.
        if capitalized:
            if 97 <= codepoint <= 122:  # to upper case.
                yield codepoint - 32

            elif 65 <= codepoint <= 90:  # keep upper case.
                yield codepoint

            else:
                yield codepoint  # maybe is nums.

            capitalized = False
            continue

        # to lower case.
        if 97 <= codepoint <= 122:  # keep lower case.
            yield codepoint

        elif 65 <= codepoint <= 90:  # to lower case.
            yield codepoint + 32

        else:
            yield codepoint  # maybe is nums.


def pipe_text_to_lower(codepoints: Iterable[int]) -> Iterator[int]:
    """
        To Lower case.
    :param codepoints:
    :return:
    """

    for codepoint in codepoints:
        if 65 <= codepoint <= 90:  # to lower case.
            yield codepoint + 32
            continue

        yield codepoint


def pipe_text_to_upper(codepoints: Iterable[int]) -> Iterator[int]:
    """
        To Upper case.
    :param codepoints:
    :return:
    """

    for codepoint in codepoints:
        if 97 <= codepoint <= 122:  # to upper case.
            yield codepoint - 32
            continue

        yield codepoint


def pipe_text_num_only(codepoints: Iterable[int], floating: bool = False) -> Iterator[int]:
    """
        Number Only.
    :param codepoints:
    :param floating:
    :return:
    """

    floated = False
    for codepoint in codepoints:

        if floating:
            if codepoint in (46,):
                if floated:  # take once.
                    break  # stop it.

                yield codepoint  # passing.
                floated = True

        if 48 <= codepoint <= 57:
            yield codepoint


def pipe_text_safe_name_kv(codepoints: Iterable[int], snippet: bool = False) -> Iterator[int]:
    """
        Skipping Another Symbol Except ASCII.
    :param codepoints:
    :param snippet:
    :return:
    """

    start = 0
    for codepoint in pipe_text_to_lower(pipe_text_safe_name(codepoints, extensible=False)):
        if codepoint == 32:  # prevent 'space' to '_'
            yield 95  # '_'
            continue

        if snippet:  # auto snippet.
            if start == 0:
                if 48 <= codepoint <= 57:  # key start with nums.
                    yield 36  # snippet char '$' before return nums.

                start = 1  # continuously.
        yield codepoint

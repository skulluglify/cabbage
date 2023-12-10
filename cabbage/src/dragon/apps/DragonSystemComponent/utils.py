#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import importlib
import importlib.machinery
import inspect
import os
import platform
import regex
import shutil
import sys

from pathlib import Path
from types import ModuleType


def makedirs(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], mode: int = 0o777, exist_ok: bool = True):
    os.makedirs(path, mode=mode, exist_ok=exist_ok)


def do_remove(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> int:
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove file or link!
            return 1

        elif os.path.isdir(path):
            shutil.rmtree(path)
            return 1  # remove by recursive!

        else:
            return -1  # exists, can't remove!

    return 0  # no exists!


def get_env_python_path() -> str:

    # return available environ!
    env_python_path = os.environ.get('DRAGON_Env_PythonPath')
    if env_python_path is not None:
        return env_python_path

    # check is windows or not!
    system = platform.system().lower()
    is_win = system.startswith('win')

    # escaping by semicolons, colons, and whitespace!
    separator = ';' if is_win else ':'
    whitespace = ' ' if is_win else '\\ '
    fn_escape_whitespace = (lambda x: regex.sub(r'(\\|)(\s+)', whitespace, x))

    # return values by separator!
    return separator.join(map(fn_escape_whitespace, sys.path))


def get_env_cache_dir() -> str:
    return os.environ.get('DRAGON_Cache_Dir', 'data/caches')


def get_env_build_pkg_dir() -> str:
    return os.environ.get('DRAGON_Build_Pkg_Dir', 'build/pkg')


def require(path: str | os.PathLike[str], package: str | None = None) -> ModuleType | None:

    # file is exist!
    if os.path.exists(path):

        # get module name!
        modulename = inspect.getmodulename(path)

        # module name found!
        if modulename is not None:

            # try import module!
            return importlib.import_module(modulename, package=package)

    # maybe path is a module namespace!
    if not os.path.isabs(path):

        # maybe module namespace having suffix!
        for suffix in importlib.machinery.all_suffixes():

            # remove suffix from module namespace!
            if path.endswith(suffix):
                path = path[0:len(path) - len(suffix)]
                break

        # catch part of path for namespace!
        p = Path(path)
        names = []

        for part in p.parts:
            part = regex.sub(r'\"|\'', '', part)
            part = regex.sub(r'\s+', '_', part)

            # verify all parts!
            if path in ('.', '..'):
                if path == '..':
                    if len(names) > 0:
                        if names[-1] != '..':
                            # remove current path dictionary on names!
                            names = names[:-1]
                        else:
                            raise Exception('unable to verify path into namespace')
                    else:
                        names.append('..')
                continue

            names.append(part)

        # make it like namespace!
        ns = '.'.join(names)

        # module namespace must be relative namespace!
        if package is not None:
            if not ns.startswith('.'):
                ns = '.' + ns

        # try import module!
        return importlib.import_module(ns, package=package)
    return None

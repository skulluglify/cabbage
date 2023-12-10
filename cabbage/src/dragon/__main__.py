#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import os
import sys

from apps.DragonSystemComponent.utils import get_env_python_path, do_remove, get_env_cache_dir, require


def main():
    if os.environ.get('DRAGON_Intercept_Main_Func') is None:
        os.environ.setdefault('DRAGON_Intercept_Main_Func', '1')
    else:
        # closing program!
        exit(1)

    os.environ.setdefault('DRAGON_Cache_Dir', 'data/caches')
    os.environ.setdefault('DRAGON_Build_Pkg_Dir', 'build/pkg')

    cwd = os.path.dirname(__file__)
    dragon_apps_path = os.path.join(cwd, 'apps')

    do_remove(get_env_cache_dir())

    sys.path.append(cwd)
    sys.path.append(dragon_apps_path)

    env_python_path = get_env_python_path()
    os.environ.setdefault('PYTHONPATH', env_python_path)
    os.environ.setdefault('DRAGON_Env_PythonPath', env_python_path)
    os.environ.setdefault('DRAGON_Sys_Init', '1')

    require('worker.py', 'apps.DjangoMainServer')


if str(__name__).upper() in ('__MAIN__',):
    main()

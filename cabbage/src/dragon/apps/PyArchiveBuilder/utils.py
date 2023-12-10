#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import os
import py_compile
import zipfile
from typing import List

from ..DragonSystemComponent.utils import get_env_build_pkg_dir


def create_pkg(pkg_name: str, pkg_path: str, pkg_build_dir: str | None = None) -> bool:
    pkg_build_dir = pkg_build_dir if pkg_build_dir is None else get_env_build_pkg_dir()
    pkg_dist = os.path.join(pkg_build_dir, pkg_name + '.pyz')

    os.makedirs(pkg_build_dir, exist_ok=True)
    if os.path.exists(pkg_path) and \
            os.path.isdir(pkg_path):

        namelist: List[str] = []
        with zipfile.ZipFile(pkg_dist, mode='w', compression=zipfile.ZIP_STORED) as stream:

            for source, _, names in os.walk(pkg_path):
                if not source.endswith('__pycache__'):
                    for name in names:

                        # virtual directory inside zipfile!
                        base_dir = os.path.relpath(source, pkg_path)

                        # create build directory!
                        file = os.path.join(source, name)
                        cdir = os.path.join(pkg_build_dir, source)
                        os.makedirs(cdir, exist_ok=True)

                        # virtual filename inside zipfile!
                        qfile = name

                        # create virtual directory inside zipfile!
                        if base_dir not in ('', '.', '..'):
                            if base_dir not in namelist:
                                namelist.append(base_dir)
                                stream.mkdir(base_dir)

                            # join virtual directory and virtual filename!
                            qfile = os.path.join(base_dir, qfile)

                        cfile = os.path.join(cdir, name)

                        # compile python script!
                        if name.endswith('.py'):
                            py_compile.compile(file, cfile + 'c')
                            stream.write(cfile + 'c', qfile + 'c')

                        # move / copy file!
                        else:
                            # shutil.copy(file, cdir)
                            stream.write(file, qfile)

        return True
    return False

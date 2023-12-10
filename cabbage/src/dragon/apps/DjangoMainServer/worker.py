#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import asyncio
import os
import platform
import sys
import tempfile
from importlib import import_module as include
from importlib.abc import FileLoader
from importlib.util import MAGIC_NUMBER
from zipimport import zipimporter

from ..DragonSystemComponent.utils import get_env_python_path, makedirs, get_env_cache_dir


async def wrapper() -> int:
    env_cache_dir = get_env_cache_dir()
    env_python_path = get_env_python_path()
    module = include('apps.DjangoMainServer.manage')
    spec = module.__spec__
    loader = spec.loader

    environ = dict(os.environ)  # copy global environ!
    environ['PYTHONPATH'] = env_python_path
    environ['DJANGO_SETTINGS_MODULE'] = 'apps.DjangoMainServer.admin.settings'
    options = ['runserver', '0.0.0.0:8080']

    if isinstance(loader, FileLoader | zipimporter):
        buffer = loader.get_data(spec.origin)
        ext = '.pyc' if buffer.startswith(MAGIC_NUMBER) else '.py'

        # make temporary directory!
        tempdir = os.path.join(env_cache_dir, 'main')
        makedirs(tempdir)

        with tempfile.NamedTemporaryFile(dir=tempdir, prefix=spec.name + '-', suffix=ext, delete=False) as stream:
            stream.truncate(0)
            stream.write(buffer)
            stream.seek(0)

            args = [sys.executable, stream.name, *options]
            process = await asyncio.create_subprocess_exec(*args, cwd=os.getcwd(), env=environ)

            print('TASK', process.pid, os.path.basename(stream.name))

            stdout, stderr = await process.communicate()
            exit_code = await process.wait()

            return exit_code
    return 1  # failure!


def main():
    env_worker_init = 'DRAGON_App_DjangoMainServer_Worker_Init'
    if os.environ.get(env_worker_init) is None:
        os.environ.setdefault(env_worker_init, '1')

        system = platform.system()
        if system.lower().startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            asyncio.set_event_loop(asyncio.ProactorEventLoop())

        else:
            asyncio.set_event_loop(asyncio.SelectorEventLoop())

        future = wrapper()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(future)
        loop.close()


if str(__name__).endswith('apps.DjangoMainServer.worker'):
    main()

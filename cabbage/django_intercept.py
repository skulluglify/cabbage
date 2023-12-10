#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import inspect
import os
import sys
from threading import Thread

import django
from importlib import import_module


def main():
    if os.environ.get('DJANGO_INTERCEPT_MAIN_FUNC') is None:
        os.environ.setdefault('DJANGO_INTERCEPT_MAIN_FUNC', '1')
    else:
        exit(1)

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cabbage.settings')
    django.setup()

    # import modules for interception!
    paths = sys.argv[0:]
    for path in paths:
        if os.path.exists(path):
            ns = inspect.getmodulename(path)
            if ns is None:
                raise Exception(f'unable to import module {ns}')

            # try import module!
            module = import_module(ns)
            fn = getattr(module, 'main')
            if fn is not None:
                if callable(fn):
                    t = Thread(target=fn, args=(), kwargs={})
                    t.start()
                    continue

                raise Exception(f'unable to call main func')
            raise Exception(f'unable to get main func')


if str(__name__).upper() in ('__MAIN__',):
    main()

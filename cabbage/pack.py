#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
from src.dragon.apps.PyArchiveBuilder.utils import create_pkg


def main():
    create_pkg('dragon', 'src/dragon', 'build/pkg')


if str(__name__).upper() in ('__MAIN__',):
    main()

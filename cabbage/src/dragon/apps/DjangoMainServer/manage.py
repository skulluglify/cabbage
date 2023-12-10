#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
import sys

from django.core.management import execute_from_command_line


def main():
    execute_from_command_line(sys.argv)


if str(__name__).upper() in ('__MAIN__',):
    main()

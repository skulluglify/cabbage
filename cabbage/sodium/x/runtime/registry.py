#!/usr/bin/env python3

import os
import sys

cwd = os.getcwd()
pwd = os.path.dirname(os.path.abspath(__file__))

if cwd not in sys.path:
    sys.path.append(cwd)

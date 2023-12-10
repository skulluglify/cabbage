#!/usr/bin/env python3

import math
from typing import Any


class unknown(Any):

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __bool__(self):
        return False

    def __str__(self):
        return 'unknown'

    def __repr__(self):
        return 'unknown'

    def __getitem__(self, item):
        return self

    def __len__(self):
        return math.inf

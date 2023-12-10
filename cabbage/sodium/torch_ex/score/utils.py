#!/usr/bin/env python3
from typing import Tuple


def avg_to_long_dist(avg_min: float, avg_max: float) -> Tuple[float, float]:

    long_dist_min = (1 / avg_max) - 1
    long_dist_max = (1 / avg_min) - 1

    return long_dist_min, long_dist_max


def long_dist_to_avg(long_dist_min: float, long_dist_max: float) -> Tuple[float, float]:

    avg_min = 1 / (long_dist_max + 1)
    avg_max = 1 / (long_dist_min + 1)

    return avg_min, avg_max


def main():

    _min = 76
    _max = 521

    _sum = _min + _max
    _avg_min = _min / _sum
    _avg_max = _max / _sum

    _long_dist_min = _min / _max
    _long_dist_max = _max / _min

    print(avg_to_long_dist(_avg_min, _avg_max))
    print((_long_dist_min, _long_dist_max))
    print(long_dist_to_avg(_long_dist_min, _long_dist_max))
    print((_avg_min, _avg_max))
    print(_long_dist_min * _max)  # _min
    print(_long_dist_max * _min)  # _max


if str(__name__).upper() in ("__MAIN__",):
    main()

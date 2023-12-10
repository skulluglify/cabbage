#!/usr/bin/env python3
import enum
import math
import os
from typing import TypeVar, Type, Tuple

import attrs

from sodium.x.runtime.wrapper import BaseClass


BaseFlowSelf = TypeVar("BaseFlowSelf", bound="BaseFlow")
BaseFlowType = Type[BaseFlowSelf]


class BaseFlow(BaseClass):
    """
        BaseFlow, Base of Flow (Wrapper).
        Like Flow Model.
    """
    pass


@attrs.define
class FlowStat:
    jobs: int
    processes: int
    chunksize: int

    def __mul__(self, n: int) -> Tuple[int, int, int]:
        """
            Manipulation, Ephemeral.
            (12, 4, 16) * 64 -> (12, 16, 64)
            (12, 16, 64) * 16 -> (12, 4, 16)
        :param n:
        :return:
        """

        chunksize = n
        processes: int
        a = min(self.chunksize, chunksize)
        b = max(self.chunksize, chunksize)
        c = b / a

        if self.chunksize < chunksize:
            processes = int(math.fabs(self.processes * c))
            chunksize = b
        else:
            processes = int(math.fabs(self.processes / c))
            chunksize = a

        return self.jobs, processes, chunksize


class NodeFlow(BaseFlow):
    """
        Node Flow (BaseModel).
    """
    @staticmethod
    def usage(p: float) -> FlowStat:
        """
            Flow Usage Manipulation.
        .. code-block:: python

            jobs, processes, chunksize = Flow.usage(0.6) * 4
        ..
        :param p:
        :return:
        """

        if 0.0 <= p <= 1.0:
            cpu_count = os.cpu_count()
            cpu_usage = int(cpu_count * p)
            return FlowStat(jobs=cpu_usage, processes=4, chunksize=16)

        raise Exception("flow usage should be between 0.0 and 1.0")


class FlowSelect(enum.Enum):
    """
        Flow Tunes, List all Tunes for Flow.
        Conflict with Asyncio,
        (FLOW_THREAD_QSYNC,
        FLOW_MP_QSYNC,
        OVERFLOW_QSYNC)
    """

    # Safe with Asyncio.
    SINGLE_RUN_SYNC = 0
    CORE_THREAD_SYNC = 1
    CORE_MP_SYNC = 2
    FLOW_THREAD_ASYNC = 3
    FLOW_MP_ASYNC = 4
    # Conflict with Asyncio.
    FLOW_THREAD_QSYNC = 5
    FLOW_MP_QSYNC = 6
    # Safe with Asyncio.
    MAX_CORE_SYNC = 7
    MAX_FLOW_QSYNC = 8
    OVERFLOW_ASYNC = 9
    # Conflict with Asyncio.
    OVERFLOW_QSYNC = 10
    # Benchmark Tune
    AUTO_BENCHMARK = 11
    AUTO_BENCHMARK_V2 = 12

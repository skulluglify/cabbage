#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TypeVar, ParamSpec, Callable, Iterable, Sequence, Tuple, List, AsyncIterator, Iterator

from sodium.x.runtime.pool.flow.process.common import Flow
from sodium.x.runtime.pool.flow.runner import FlowRunner
from sodium.x.runtime.pool.flow.types import NodeFlow

_T = TypeVar("_T")
_VT = TypeVar("_VT")
_P = ParamSpec("_P")
OverflowSelf = TypeVar("OverflowSelf", bound="Overflow")


class Overflow(NodeFlow):
    """
        Overflow (MP,Thread,QSync), Based On Flow (MP|Thread,QSync).
    """

    jobs: int
    processes: int
    chunksize: int

    def __init__(self, jobs: int, processes: int, chunksize: int):

        self.jobs = jobs
        self.processes = processes
        self.chunksize = chunksize

    @staticmethod
    def wrapper_qsync(func: Callable[_P, _VT],
                      params: Iterable[Sequence[_T]],
                      jobs: int,
                      chunksize: int) -> Tuple[_VT, ...]:
        """
            Overflow Wrapper, for Overflow Main Function.
        :param func:
        :param params:
        :param jobs:
        :param chunksize:
        :return:
        """

        outputs: List[_VT] = []
        with Flow(executor=ThreadPoolExecutor, jobs=jobs, chunksize=chunksize) as flow:
            for result in flow.map_qsync(func, params):
                outputs.append(result)

        return tuple(outputs)

    async def map(self,
                  func: Callable[_P, _VT],
                  params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT]:
        """
            Overflow, Make it Combine Process Pools and Thread Pools.
        :param func:
        :param params:
        :return:
        """

        params: Tuple[Sequence[_T], ...] = tuple(params)
        n_params = len(params)

        # Cause Chunk on Params, mp_chunksize is the same as standalone process.
        mp_chunksize = self.processes

        with Flow(executor=ProcessPoolExecutor, jobs=self.jobs, chunksize=mp_chunksize) as flow:

            # Chunk for Params.
            x_params: List[Tuple[Callable[_P, _VT], Iterable[Sequence[_T]], int, int]] = []
            m_process = max(int(n_params / self.processes), 1)

            for i in range(0, n_params, m_process):
                k = min(i + m_process, n_params)
                x_params.append((func, params[i:k], self.jobs, self.chunksize))

            outputs: Tuple[_VT, ...]
            async for outputs in flow.map(self.wrapper_qsync, x_params):
                for result in outputs:
                    yield result

    def map_qsync(self,
                  func: Callable[_P, _VT],
                  params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            Overflow, Make it Combine Process Pools and Thread Pools.
        :param func:
        :param params:
        :return:
        """

        runner = FlowRunner(self.map(func, params))
        return runner.sync()

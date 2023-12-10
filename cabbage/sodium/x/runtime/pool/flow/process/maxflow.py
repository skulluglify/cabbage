#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TypeVar, ParamSpec, Tuple, Callable, Iterable, Sequence, List, Iterator

from sodium.x.runtime.pool.flow.types import NodeFlow
from sodium.x.runtime.pool.flow.process.common import Flow

_T = TypeVar("_T")
_VT = TypeVar("_VT")
_P = ParamSpec("_P")
MaxFlowSelf = TypeVar("MaxFlowSelf", bound="MaxFlow")


class MaxFlow(NodeFlow):
    """
        MaxFlow (MP,Thread,QSync), Based On CORE_MP, And Flow.
    """

    jobs: int
    processes: int
    chunksize: int

    def __init__(self, jobs: int, processes: int, chunksize: int):

        self.jobs = jobs
        self.processes = processes
        self.chunksize = chunksize

    @classmethod
    def _apply_wrapper_qsync(cls, data: Tuple[Callable[_P, _VT], Iterable[_T], int, int]) -> Tuple[_VT, ...]:
        """
            Apply Wrapper, Unpack Single Parameter into Parameters.
        :param data:
        :return:
        """

        func, params, jobs, chunksize = data
        return cls.wrapper_qsync(func, params, jobs, chunksize)

    @staticmethod
    def wrapper_qsync(func: Callable[_P, _VT],
                      params: Iterable[Sequence[_T]],
                      jobs: int,
                      chunksize: int) -> Tuple[_VT, ...]:
        """
            Wrapper, for Multi Process.
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

    def map(self,
            func: Callable[_P, _VT],
            params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            MaxFlow, Make it Combine Process Pools and Thread Pools.
        :param func:
        :param params:
        :return:
        """

        params: Tuple[Sequence[_T], ...] = tuple(params)
        n_params = len(params)
        mp_chunksize = 1

        with ProcessPoolExecutor(max_workers=self.processes) as pool:

            x_params: List[Tuple[Callable[_P, _VT], Iterable[Sequence[_T]], int, int]] = []
            m_process = max(int(n_params / self.processes), 1)

            for i in range(0, n_params, m_process):
                k = min(i + m_process, n_params)
                x_params.append((func, params[i:k], self.jobs, self.chunksize))

            outputs: Tuple[_VT, ...]

            for outputs in pool.map(self._apply_wrapper_qsync, x_params, chunksize=mp_chunksize):

                for result in outputs:
                    yield result

        return

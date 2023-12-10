#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TypeVar, Tuple, Callable, Iterable, Sequence, List, Iterator, ParamSpec

from sodium.x.runtime.pool.flow.types import NodeFlow

_T = TypeVar("_T")
_VT = TypeVar("_VT")
_P = ParamSpec("_P")
MaxCoreSelf = TypeVar("MaxCoreSelf", bound="MaxCore")


class MaxCore(NodeFlow):
    """
        MaxCore (MP,Thread,Sync), Based On CORE_MP, And CORE_THREAD.
    """

    jobs: int
    processes: int
    chunksize: int

    def __init__(self, jobs: int, processes: int, chunksize: int = 1):
        """
            MaxCore, Chunksize is not required. (for now)
        :param jobs:
        :param processes:
        :param chunksize:
        """

        self.jobs = jobs
        self.processes = processes
        self.chunksize = chunksize

    @classmethod
    def _apply_wrapper(cls, data: Tuple[Callable[_P, _VT], Iterable[_T], int]) -> Tuple[_VT, ...]:
        """
            Apply Wrapper, Unpack Single Parameter into Parameters.
        :param data:
        :return:
        """

        func, params, jobs = data
        return cls.wrapper(func, params, jobs)

    @staticmethod
    def wrapper(func: Callable[_P, _VT],
                params: Iterable[Sequence[_T]],
                jobs: int) -> Tuple[_VT, ...]:
        """
            Wrapper, for Multi Process.
        :param func:
        :param params:
        :param jobs:
        :return:
        """

        outputs: List[_VT] = []

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for result in pool.map(func, params):
                outputs.append(result)

        return tuple(outputs)

    def map(self,
            func: Callable[_P, _VT],
            params: Iterable[_T]) -> Iterator[_VT]:
        """
            MaxCore, Make it Combine Process Pools and Thread Pools.
        :param func:
        :param params:
        :return:
        """

        params: Tuple[_T, ...] = tuple(params)
        n_params = len(params)
        mp_chunksize = 1

        with ProcessPoolExecutor(max_workers=self.processes) as pool:

            x_params: List[Tuple[Callable[_P, _VT], Iterable[_T], int]] = []
            m_process = max(int(n_params / self.processes), 1)

            for i in range(0, n_params, m_process):
                k = min(i + m_process, n_params)
                x_params.append((func, params[i:k], self.jobs))

            outputs: Tuple[_VT, ...]

            for outputs in pool.map(self._apply_wrapper, x_params, chunksize=mp_chunksize):

                for result in outputs:
                    yield result

        return

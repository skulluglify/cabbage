#!/usr/bin/env python3
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TypeVar, ParamSpec, Type, Iterable, Sequence, Iterator, Tuple, Callable, AsyncIterable, AsyncIterator

from sodium.x.runtime.pool.flow.process.common import Flow
from sodium.x.runtime.pool.flow.process.maxcore import MaxCore
from sodium.x.runtime.pool.flow.process.maxflow import MaxFlow
from sodium.x.runtime.pool.flow.process.overflow import Overflow
from sodium.x.runtime.pool.flow.runner import FlowRunner
from sodium.x.runtime.pool.flow.types import BaseFlow, FlowSelect

_T = TypeVar("_T")
_VT = TypeVar("_VT")
_P = ParamSpec("_P")
FlowContextSelf = TypeVar("FlowContextSelf", bound="FlowContext")
FlowContextType = Type[FlowContextSelf]


class FlowContext(BaseFlow):
    """
        FlowContext, Flow Manager.
    """

    jobs: int
    processes: int
    chunksize: int
    tune: FlowSelect
    logger: logging.Logger

    def __init__(self, jobs: int, processes: int, chunksize: int, tune: FlowSelect = FlowSelect.MAX_CORE_SYNC):

        self.logger = logging.getLogger(self.name)

        self.jobs = jobs
        self.processes = processes
        self.chunksize = chunksize
        self.tune = tune

    @staticmethod
    def seq_params(params: Iterable[Sequence[_T] | _T], seq: bool = True) -> Iterator[Sequence[_T]]:
        """
            Sequence Parameters, Sequence Parameters in Function with Single Parameter.
            ex. Function([...Parameters])
            if not set Sequence, will be
            ex. Function(...Parameters)
        :param params:
        :param seq:
        :return:
        """

        if seq:
            return map(lambda param: (param,), params)

        return iter(params)

    @staticmethod
    def wrapper_unpack_params_on_func(data: Tuple[Callable[_P, _VT], Sequence[_T]]) -> _VT:
        """
            Wrapper Support Unpack Params on Concurrent Futures.
        :param data:
        :return:
        """

        func: Callable[_P, _VT]
        params: Sequence[_T]
        func, params = data

        if callable(func):
            if isinstance(params, Sequence):
                return func(*params)

            raise RuntimeError("parameters is not sequenced")

        raise RuntimeError("function is not callable")

    @staticmethod
    def _hook_func_on_params(func: Callable[_P, _VT],
                             params: Iterable[Sequence[_T]]) -> Iterable[Tuple[Callable[_P, _VT], Sequence[_T]]]:
        """
            Using Wrapper Unpack Params, Must be Used Wrapper Hook Function On Params,
            Cause Wrapper Unpack Params Needed Function Callback.
        :param func:
        :param params:
        :return:
        """

        for param in params:
            yield func, param

    def auto_tune_diversion(self, tune: FlowSelect) -> FlowSelect:
        """
            Auto Tune Switcher for Flow.
        :param tune:
        :return:
        """

        if tune in (FlowSelect.AUTO_BENCHMARK, FlowSelect.AUTO_BENCHMARK_V2):
            raise Exception("tune is not accepted for main context")

        if self.jobs <= 0:
            if self.jobs == 0:
                if tune not in (FlowSelect.SINGLE_RUN_SYNC,):
                    tune = FlowSelect.SINGLE_RUN_SYNC
                    self.logger.warning(f"switch flow tune to {tune.name}, cause jobs is 0")
            else:
                raise RuntimeError("jobs must be greater than or equal 0")

        if self.processes <= 0:
            if self.processes == 0:
                if tune not in (FlowSelect.OVERFLOW_ASYNC,
                                FlowSelect.OVERFLOW_QSYNC,
                                FlowSelect.MAX_FLOW_QSYNC,
                                FlowSelect.MAX_CORE_SYNC):

                    if tune in (FlowSelect.OVERFLOW_ASYNC,):
                        tune = FlowSelect.FLOW_THREAD_ASYNC

                    elif tune in (FlowSelect.MAX_FLOW_QSYNC, FlowSelect.OVERFLOW_QSYNC):
                        tune = FlowSelect.FLOW_THREAD_QSYNC

                    else:
                        tune = FlowSelect.CORE_THREAD_SYNC

                    self.logger.warning(f"switch flow tune to {tune.name}, cause processes is 0")
            else:
                raise RuntimeError("processes must be greater than or equal 0")

        if self.chunksize <= 0:
            if self.chunksize == 0:
                if tune not in (FlowSelect.FLOW_THREAD_ASYNC,
                                FlowSelect.FLOW_THREAD_QSYNC,
                                FlowSelect.FLOW_MP_ASYNC,
                                FlowSelect.FLOW_MP_QSYNC,
                                FlowSelect.MAX_FLOW_QSYNC):

                    if tune in (FlowSelect.FLOW_THREAD_ASYNC, FlowSelect.FLOW_THREAD_QSYNC):
                        tune = FlowSelect.CORE_THREAD_SYNC

                    elif tune in (FlowSelect.FLOW_MP_ASYNC, FlowSelect.FLOW_MP_QSYNC):
                        tune = FlowSelect.CORE_MP_SYNC

                    else:
                        tune = FlowSelect.MAX_CORE_SYNC

                    self.logger.warning(f"switch flow tune to {tune.name}, cause processes is 0")
            else:
                raise RuntimeError("chunksize must be greater than or equal 0")

        return tune

    def run(self,
            func: Callable[_P, _VT],
            params: Iterable[_T],
            unpack: bool = False,
            tune: FlowSelect | None = None) -> FlowRunner:

        map_co_iter: AsyncIterable[_VT] | Iterable[_VT]
        seq = not unpack

        if tune is None:
            tune = self.tune

        tune = self.auto_tune_diversion(tune)

        if tune == FlowSelect.CORE_THREAD_SYNC:
            if unpack:
                map_co_iter = self.fp_core_sync_thread(self.wrapper_unpack_params_on_func,
                                                       self._hook_func_on_params(func, params))

            else:
                map_co_iter = self.fp_core_sync_thread(func, params)

        elif tune == FlowSelect.CORE_MP_SYNC:
            if unpack:
                # raise RuntimeError(f"unpack params on {tune.name} is not supported")
                map_co_iter = self.fp_core_sync_mp(self.wrapper_unpack_params_on_func,
                                                   self._hook_func_on_params(func, params))

            else:
                map_co_iter = self.fp_core_sync_mp(func, params)

        elif tune == FlowSelect.FLOW_THREAD_ASYNC:
            map_co_iter = self.fp_flow_async_thread_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.FLOW_MP_ASYNC:
            map_co_iter = self.fp_flow_async_mp_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.FLOW_THREAD_QSYNC:
            map_co_iter = self.fp_flow_qsync_thread_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.FLOW_MP_QSYNC:
            map_co_iter = self.fp_flow_qsync_mp_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.MAX_CORE_SYNC:
            if unpack:
                map_co_iter = self.fp_max_core_sync(self.wrapper_unpack_params_on_func,
                                                    self._hook_func_on_params(func, params))

            else:
                map_co_iter = self.fp_max_core_sync(func, params)

        elif tune == FlowSelect.MAX_FLOW_QSYNC:
            map_co_iter = self.fp_max_flow_qsync_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.OVERFLOW_ASYNC:
            map_co_iter = self.fp_overflow_async_unpack_params(func, self.seq_params(params, seq))

        elif tune == FlowSelect.OVERFLOW_QSYNC:
            map_co_iter = self.fp_overflow_qsync_unpack_params(func, self.seq_params(params, seq))

        else:
            if unpack:
                def wrapper(__params: Sequence[_T]) -> _VT:
                    """
                        Unpack Parameters, Passing Parameters in Function.
                    :param __params:
                    :return:
                    """

                    return func(*__params)

                # if params is Iterable[Sequence[_T]]
                map_co_iter = map(wrapper, params)

            else:
                map_co_iter = map(lambda param: func(param), params)

        return FlowRunner(map_co_iter)

    def fp_core_sync_thread(self,
                            func: Callable[_P, _VT],
                            params: Iterable[_T]) -> Iterator[_VT]:
        """
            FuncProc, Core Thread.
        :param func:
        :param params:
        :return:
        """

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for result in pool.map(func, params):
                yield result

    def fp_core_sync_mp(self,
                        func: Callable[_P, _VT],
                        params: Iterable[_T]) -> Iterator[_VT]:
        """
            FuncProc, Core Multi Process.
        :param func:
        :param params:
        :return:
        """

        with ProcessPoolExecutor(max_workers=self.jobs) as pool:
            for result in pool.map(func, params):
                yield result

    async def fp_flow_async_thread_unpack_params(self,
                                                 func: Callable[_P, _VT],
                                                 params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT]:
        """
            FuncProc, Flow Thread.
        :param func:
        :param params:
        :return:
        """

        with Flow(executor=ThreadPoolExecutor, jobs=self.jobs, chunksize=self.chunksize) as flow:
            async for result in flow.map(func, params):
                yield result

    async def fp_flow_async_mp_unpack_params(self,
                                             func: Callable[_P, _VT],
                                             params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT]:
        """
            FuncProc, Flow Multi Process.
        :param func:
        :param params:
        :return:
        """

        with Flow(executor=ProcessPoolExecutor, jobs=self.jobs, chunksize=self.chunksize) as flow:
            async for result in flow.map(func, params):
                yield result

    def fp_flow_qsync_thread_unpack_params(self,
                                           func: Callable[_P, _VT],
                                           params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            FuncProc, Flow Thread (QSync).
        :param func:
        :param params:
        :return:
        """

        with Flow(executor=ThreadPoolExecutor, jobs=self.jobs, chunksize=self.chunksize) as flow:
            for result in flow.map_qsync(func, params):
                yield result

    def fp_flow_qsync_mp_unpack_params(self,
                                       func: Callable[_P, _VT],
                                       params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            FuncProc, Flow Multi Process (QSync).
        :param func:
        :param params:
        :return:
        """

        with Flow(executor=ProcessPoolExecutor, jobs=self.jobs, chunksize=self.chunksize) as flow:
            for result in flow.map_qsync(func, params):
                yield result

    def fp_max_core_sync(self,
                         func: Callable[_P, _VT],
                         params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            FuncProc, MaxCore.
        :param func:
        :param params:
        :return:
        """

        with MaxCore(jobs=self.jobs, processes=self.processes, chunksize=self.chunksize) as overflow:
            for result in overflow.map(func, params):
                yield result

    def fp_max_flow_qsync_unpack_params(self,
                                        func: Callable[_P, _VT],
                                        params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            FuncProc, MaxFlow.
        :param func:
        :param params:
        :return:
        """

        with MaxFlow(jobs=self.jobs, processes=self.processes, chunksize=self.chunksize) as overflow:
            for result in overflow.map(func, params):
                yield result

    async def fp_overflow_async_unpack_params(self,
                                              func: Callable[_P, _VT],
                                              params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT]:
        """
            FuncProc, Overflow.
        :param func:
        :param params:
        :return:
        """

        with Overflow(jobs=self.jobs, processes=self.processes, chunksize=self.chunksize) as overflow:
            async for result in overflow.map(func, params):
                yield result

    def fp_overflow_qsync_unpack_params(self,
                                        func: Callable[_P, _VT],
                                        params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            FuncProc, Overflow (QSync).
        :param func:
        :param params:
        :return:
        """

        with Overflow(jobs=self.jobs, processes=self.processes, chunksize=self.chunksize) as overflow:
            for result in overflow.map_qsync(func, params):
                yield result

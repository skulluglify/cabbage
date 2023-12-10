#!/usr/bin/env python3
from asyncio import Lock, Future
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Executor
from typing import TypeVar, Type, Callable, AsyncIterator, Tuple, Sequence, Iterable, Iterator, ParamSpec, List

from sodium.x.runtime.pool.flow.runner import FlowRunner
from sodium.x.runtime.pool.flow.types import NodeFlow

_T = TypeVar("_T")
_VT = TypeVar("_VT")
_P = ParamSpec("_P")
FlowSelf = TypeVar("FlowSelf", bound="Flow")


class Flow(NodeFlow):
    """
        Flow (MP|Thread,QSync).
        Handle Multiple process without waiting first order.
    """

    chunksize: int
    executor: Type[ProcessPoolExecutor | ThreadPoolExecutor]
    mutex: Lock
    tasks: List[Future | None]
    jobs: int

    def __init__(self, executor: Type[ProcessPoolExecutor | ThreadPoolExecutor], jobs: int, chunksize: int = 64):

        self.chunksize = chunksize
        self.executor = executor
        self.mutex = Lock()
        self.tasks = [None] * chunksize
        self.jobs = jobs

    def detach_nowait(self) -> bool:

        found = False
        for task in self.tasks:
            if task is None:
                found = True
                break

        return found

    async def detach(self) -> bool:

        await self.mutex.acquire()
        result = self.detach_nowait()
        self.mutex.release()
        return result

    def empty_nowait(self) -> bool:

        found = False
        for task in self.tasks:
            if task is not None:
                found = True
                break

        return not found

    async def empty(self) -> bool:

        await self.mutex.acquire()
        result = self.empty_nowait()
        self.mutex.release()
        return result

    def put_nowait(self, future: Future[Callable[[_P], _VT]]) -> bool:

        while True:
            if len(self.tasks) != 0:

                i: int
                task: Future[Callable[[_P], _VT]]
                found = False

                for i, task in enumerate(self.tasks):
                    if task is None:
                        self.tasks[i] = future
                        found = True
                        break

                if found:
                    return True

            return False

    async def put(self, future: Future[Callable[[_P], _VT]]) -> bool:

        await self.mutex.acquire()
        result = self.put_nowait(future)
        self.mutex.release()
        return result

    async def wait(self, check_all: bool = True, wait_all: bool = False) -> AsyncIterator[_VT]:

        await self.mutex.acquire()

        while True:

            if self.empty_nowait():
                break

            i: int
            task: Future[Callable[[_P], _VT]]
            found = False

            for i, task in enumerate(self.tasks):

                if task is None:
                    continue

                if task.done():
                    found = True

                    # Skipping if it cancelled.
                    if task.cancelled():
                        self.tasks[i] = None
                        continue

                    # Result.
                    self.tasks[i] = None
                    yield task.result()

                    # Stop if Not Check All.
                    if not check_all:
                        break

            if not wait_all:
                if found:
                    break

        self.mutex.release()

    def qsize(self) -> int:

        n = 0
        for task in self.tasks:
            if task is None:
                continue

            n += 1

        return n

    def fill_nowait(self,
                    pool: Executor,
                    func: Callable[_P, _VT],
                    params: Tuple[Sequence[_T], ...]) -> Tuple[int, Tuple[Sequence[_T], ...]]:

        n_params = len(params)
        # idle_task_size = self.chunksize - self.qsize()
        j = 0
        for i, task in enumerate(self.tasks):
            if n_params <= j:
                break

            if task is not None:
                continue

            param = params[j]
            self.tasks[i] = pool.submit(func, *param)
            j += 1

        return j, params[j:]

    async def fill(self,
                   pool: Executor,
                   func: Callable[_P, _VT],
                   params: Tuple[Sequence[_T], ...]) -> Tuple[int, Tuple[Sequence[_T], ...]]:

        await self.mutex.acquire()
        result = self.fill_nowait(pool, func, params)
        self.mutex.release()
        return result

    async def map(self, func: Callable[_P, _VT], params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT]:

        with self.executor(max_workers=self.jobs) as pool:

            # Casting into Tuple.
            params = tuple(params)

            n_params = len(params)
            m_params = 0

            while True:

                if len(params) != 0:
                    k, params = await self.fill(pool, func, params)

                i = 0
                async for result in self.wait():
                    yield result
                    i += 1

                m_params += i

                if n_params <= m_params:
                    break

    def map_qsync(self, func: Callable[_P, _VT], params: Iterable[Sequence[_T]]) -> Iterator[_VT]:
        """
            Map Iteration with Queue Synchronization.
        :param func:
        :param params:
        :return:
        """

        runner = FlowRunner(self.map(func, params))
        return runner.sync()

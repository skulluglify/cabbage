#!/usr/bin/env python3
import asyncio
from asyncio import AbstractEventLoop
from typing import Iterable, AsyncIterable, Iterator, AsyncIterator, Callable, TypeVar, Type

from sodium.x.runtime.pool.flow.types import BaseFlow


_T = TypeVar("_T")
_VT = TypeVar("_VT")
FlowRunnerSelf = TypeVar("FlowRunnerSelf", bound="FlowRunner")
FlowRunnerType = Type[FlowRunnerSelf]


class FlowRunner(BaseFlow):
    """
        FlowRunner, Flowing Runner.
        Running Flow with Async / Sync.
    """

    _map_co_iter__: AsyncIterable[_T] | Iterable[_T]

    def __init__(self, iterable: AsyncIterable[_T] | Iterable[_T]):
        self._map_co_iter__ = iterable

    def __iter__(self) -> Iterator[_T]:
        """
            Passing Use Iter Function.
        :return:
        """

        if isinstance(self._map_co_iter__, Iterable):
            return self._wrapper_iter(self._map_co_iter__)

        return self.sync()

    def __aiter__(self) -> AsyncIterator[_T]:
        """
            Passing Use Aiter Function.
        :return:
        """

        if isinstance(self._map_co_iter__, AsyncIterable):
            return self._wrapper_async_iter(self._map_co_iter__)

        raise RuntimeWarning("choice QSYNC not suitable for asynchronously, "
                             "make sure not using QSYNC, understanding warning message, "
                             "used make_async instead of aiter")
        # return self.make_async()

    @staticmethod
    async def _wrapper_async_iter(map_co_iter: AsyncIterable[_T]) -> AsyncIterator[_T]:
        """
            Wrapping AsyncIterable into AsyncIterator.
        :param map_co_iter:
        :return:
        """
        async for result in map_co_iter:
            yield result

    @staticmethod
    def _wrapper_iter(map_iter: Iterable[_T]) -> Iterator[_T]:
        """
            Wrapping Iterable into Iterator.
        :param map_iter:
        :return:
        """
        for result in map_iter:
            yield result

    async def _map_async(self, cb: Callable[[_T, int], _VT]) -> AsyncIterator[_VT]:
        """
            Wrapping Map Async, Return AsyncIterator.
        :param cb:
        :return:
        """
        i = 0
        async for result in self._map_co_iter__:
            yield cb(result, i)
            i += 1

    @classmethod
    def create(cls: FlowRunnerType, iterable: AsyncIterable[_T] | Iterable[_T]) -> FlowRunnerSelf:
        """
            Create New FlowRunner.
        :param iterable:
        :return:
        """

        Wrapper = cls
        return Wrapper(iterable)

    def map(self, cb: Callable[[_T, int], _VT]) -> FlowRunnerSelf:
        """
            Map Iterator, Async / Iterable.
        :param cb:
        :return:
        """

        return self.create(self._map_async(cb))

    async def make_async(self) -> AsyncIterator[_T]:
        """
            Make Async / Sync to Async.
        Warning for QSYNC, not suitable for async.
        :return:
        """

        if isinstance(self._map_co_iter__, AsyncIterable):
            async for result in self._map_co_iter__:
                yield result

        if isinstance(self._map_co_iter__, Iterable):
            for result in self._map_co_iter__:
                yield result

        return

    def sync(self) -> Iterator[_T]:
        """
            Make Async / QSync / Sync to Sync.
        :return:
        """

        event: AbstractEventLoop
        if isinstance(self._map_co_iter__, AsyncIterable):
            event = self.hook_event_loop()
            map_co_iter = self._wrapper_async_iter(self._map_co_iter__)

            while True:
                try:
                    map_co_next = anext(map_co_iter)
                    result = event.run_until_complete(map_co_next)
                    yield result

                except StopAsyncIteration:
                    break

            return

        # If is Iterable.
        if isinstance(self._map_co_iter__, Iterable):
            for result in self._map_co_iter__:
                yield result

        return

    @staticmethod
    def hook_event_loop() -> AbstractEventLoop:
        """
            Hook or Create New One.
        :return:
        """

        try:
            # create new one
            return asyncio.new_event_loop()

        # RuntimeError: Cannot run the event loop while another loop is running
        except RuntimeError:

            # hook already build
            event = asyncio.get_event_loop()

            # waiting, forever.
            while event.is_running():
                pass

            return event

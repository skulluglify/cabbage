#!/usr/bin/env python3
from typing import Protocol, ParamSpec, TypeVar, Callable, Iterable, Sequence, AsyncIterator, Iterator

# TODO: Implemented ProtoFlow to All Models.
# TODO: Added MaxFlow2 for Async (FLOW_MP, CORE_THREAD).
_P = ParamSpec("_P")
_T = TypeVar("_T")
_VT = TypeVar("_VT")


class ProtoFlow(Protocol):

    async def map(self, func: Callable[_P, _VT], params: Iterable[Sequence[_T]]) -> AsyncIterator[_VT] | Iterator[_VT]:
        pass

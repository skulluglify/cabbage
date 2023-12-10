#!/usr/bin/env python3
import math
from functools import partial
from typing import Callable, ParamSpec, TypeVar, Any, Type, Tuple, Dict, List, Generic

import attrs

from sodium.x.runtime.utils.refs import TypeAny

_P = ParamSpec("_P")
_VT = TypeVar("_VT")

StateSelf = TypeVar("StateSelf", bound="State")
StateType = Type[StateSelf]


@attrs.define
class State(Generic[_VT]):

    ref: Any
    name: str
    value: _VT

    # TODO: Implemented State Refresh Than State Clear.
    revoke: bool


@attrs.define
class StateStoreValues:
    """
        State Store Values.
    """

    states: List[State] = []

    def get_state(self, ref: Any, name: str) -> State | None:
        for state in self.states:
            if state.ref == ref:
                if state.name == name:
                    return state

        return None

    def get_state_by_type(self, ref_type: TypeAny, name: str) -> State | None:
        for state in self.states:
            if type(state.ref) is ref_type:
                if state.name == name:
                    return state

        return None

    def del_state(self, ref: Any, name: str) -> bool:
        state = self.get_state(ref=ref, name=name)
        if state is None:
            return False

        self.states.remove(state)
        return True

    def del_state_by_type(self, ref_type: TypeAny, name: str) -> bool:
        state = self.get_state_by_type(ref_type=ref_type, name=name)
        if state is None:
            return False

        self.states.remove(state)
        return True

    def find_state(self, ref: Any, name: str) -> bool:
        return self.get_state(ref=ref, name=name) is not None

    def find_state_by_type(self, ref_type: TypeAny, name: str) -> bool:
        return self.get_state_by_type(ref_type=ref_type, name=name) is not None

    # Must have Ref Of Value.
    def push_state(self, state: State):
        self.states.append(state)

    # Must have Ref Of Value.
    def put_state(self, ref: Any, name: str, value: Any) -> bool:

        # No Duplicated.
        if self.find_state(ref=ref, name=name):
            return False

        # Push.
        state = State(ref=ref, name=name, value=value, revoke=False)
        self.push_state(state)
        return True


FuncStateHelperSelf = TypeVar("FuncStateHelperSelf", bound="FuncStateHelper")
FuncStateHelperType = Type[FuncStateHelperSelf]


class FuncStateHelper(partial):
    """
        Func State Wrapper for Func State.
    """

    _func: Callable[_P, _VT]
    _refresh: Callable[[], Any]
    _reset: Callable[[], Any]
    _params: Tuple[Any, ...]
    _dict_params: Dict[str, Any]

    def __new__(cls: FuncStateHelperType,
                func: Callable[_P, _VT],
                params: Tuple[Any, ...] | None = None,
                dict_params: Dict[str, Any] | None = None,
                refresh: Callable[[], Any] = (lambda: None),
                reset: Callable[[], Any] = (lambda: None)):

        if params is None:
            params = tuple()

        if dict_params is None:
            dict_params = dict()

        base = super(FuncStateHelper, cls)
        self = base.__new__(cls, func, *params, **dict_params)
        self._refresh = refresh
        self._reset = reset
        return self

    def refresh(self: FuncStateHelperSelf):
        if not callable(self._refresh):
            raise Exception("Not implemented func 'refresh'")
        self._refresh()

    def reset(self: FuncStateHelperSelf):
        if not callable(self._reset):
            raise Exception("Not implemented func 'reset'")
        self._reset()


FuncStateSelf = TypeVar("FuncStateSelf", bound="FuncState")
FuncStateType = Type[FuncStateSelf]


class FuncState:
    """
        Func State.
    """

    cached: bool
    func: Callable[_P, _VT]
    callback: Callable[_P, _VT] | None
    check_output: bool
    output: _VT | None
    keep: bool

    # For Instance Methods. (Object Classes)
    state_store_values: StateStoreValues

    def __init__(self: FuncStateSelf,
                 func: Callable[_P, _VT],
                 callback: Callable[[_VT], Any] | None = None,
                 cb: Callable[[_VT], Any] | None = None,
                 check_output: bool = False,
                 v: bool = False,
                 keep: bool = False):
        """
            Function State Wrapper
        Examples:

            >>> # func: Function, callback: Function Callback
            >>> FuncState(func, callback=cb)
        :param func:
        :param callback:
        :param cb:
        :param check_output:
        :param v:
        """

        # Short Assigns.
        callback = callback or cb
        check_output = check_output or v

        self.cached = False
        self.callback = callback
        self.check_output = check_output
        self.func = func
        self.output = None
        self.keep = keep

        # For Instance Methods. (Object Classes)
        self.state_store_values = StateStoreValues()

    @staticmethod
    def _check_bool_is_false(val: Any) -> bool:
        """
            Check Value is Boolean.
        :param val:
        :return:
        """

        if isinstance(val, bool):
            return not val

        return False

    @staticmethod
    def _check_num_is_inf_nan_zero(val: Any) -> bool:
        """
            Check is Inf Or Nan.
        :param val:
        :return:
        """
        if not isinstance(val, complex) or \
                not isinstance(val, float) or \
                not isinstance(val, int):
            return False

        num: complex | float | int
        num = val

        if isinstance(num, complex):
            num = num.real

        if isinstance(num, float):
            if math.isnan(num) or math.isinf(num) or num in (0.0,):
                return True

        if isinstance(num, int):
            if math.isnan(num) or num in (0,):
                return True

        return False

    @staticmethod
    def _check_str_is_empty(val: Any) -> bool:
        """
            Check Value is Empty String.
        :param val:
        :return:
        """

        if isinstance(val, str):
            return val.strip() == ""

        return False

    def __call__(self: FuncStateSelf, *args: _P.args, **kwargs: _P.kwargs) -> _VT:
        """
            Function Wrapper.
        :param args:
        :param kwargs:
        :return:
        """

        if not self.cached:
            self.output = self.func(*args, **kwargs)
            self.cached = True

        if self.check_output:

            # Check Type Output is Boolean
            if self._check_bool_is_false(self.output):
                self.output = self.func(*args, **kwargs)

            # Check Type Output is Number
            if self._check_num_is_inf_nan_zero(self.output):
                self.output = self.func(*args, **kwargs)

            # Check Type Output is String
            if self._check_str_is_empty(self.output):
                self.output = self.func(*args, **kwargs)

        if callable(self.callback):
            self.callback(self.output)

        return self.output

    def _state_call(self: FuncStateSelf,
                    objects: Tuple[Any, Type[Any] | None],
                    *args: _P.args, **kwargs: _P.kwargs) -> _VT:

        instance, owner = objects

        func = self.func
        func_name = func.__qualname__
        state_store_values = self.state_store_values

        state: State
        if not self.keep:
            state = state_store_values.get_state(ref=instance, name=func_name)

        else:
            state = state_store_values.get_state_by_type(ref_type=owner, name=func_name)

        if state is None:

            result = func(*args, *kwargs)
            if self.check_output:

                # Check Type Output is Boolean
                if self._check_bool_is_false(result):
                    result = func(*args, **kwargs)

                # Check Type Output is Number
                if self._check_num_is_inf_nan_zero(result):
                    result = func(*args, **kwargs)

                # Check Type Output is String
                if self._check_str_is_empty(result):
                    result = func(*args, **kwargs)

            state = State(ref=instance, name=func_name, value=result, revoke=False)
            state_store_values.push_state(state)

        if callable(self.callback):
            self.callback(state.value)

        return state.value

    def _state_clear(self: FuncStateSelf, objects: Tuple[Any, Type[Any] | None]):

        instance, owner = objects

        func = self.func
        func_name = func.__qualname__
        state_store_values = self.state_store_values

        if not self.keep:
            state_store_values.del_state(ref=instance, name=func_name)

        else:
            state_store_values.del_state_by_type(ref_type=owner, name=func_name)

    def __get__(self: FuncStateSelf, instance: Any, owner: Type[Any] | None = None) -> FuncStateHelper:
        """
            Support Instance Methods.
        :param instance:
        :param owner:
        :return:
        """

        func = self._state_call
        params = ((instance, owner), instance)
        state_clear = (lambda: self._state_clear((instance, owner)))

        return FuncStateHelper(func, params=params, refresh=state_clear, reset=state_clear)

    def __set__(self: FuncStateSelf, instance: Any, value: Any):
        """
            Func State Protected.
        :param instance:
        :param value:
        :return:
        """
        func_state_name = self.func.__name__
        raise Exception(f"Method '{func_state_name}' has been restricted")

    # def __del__(self): ...

    def refresh(self: FuncStateSelf):
        """
            Refresh, Keep Outputs.
        :return:
        """
        self.cached = False

    def reset(self: FuncStateSelf):
        """
            Reset Hard.
        :return:
        """
        self.cached = False
        self.output = None


def func_state(func: Callable[_P, _VT] | None = None,
               callback: Callable[[_VT], Any] | None = None,
               cb: Callable[[_VT], Any] | None = None,
               check_output: bool = False,
               v: bool = False,
               keep: bool = False) -> FuncState | Callable[[Callable[_P, _VT]], FuncState]:
    """
        Function State.

    Examples:

        >>> @func_state
        >>> def first_call_only(x: int):
        >>>     ...
        >>>
        >>> @func_state(cb=lambda x: print(x))
        >>> def first_call_only(x: int):
        >>>     ...
        >>>
    :param func:
    :param callback:
    :param cb:
    :param check_output:
    :param v:
    :param keep:
    :return:
    """

    # Short Assigns.
    callback = callback or cb
    check_output = check_output or v

    if func is None:
        return partial(FuncState, callback=callback, check_output=check_output, keep=keep)

    return FuncState(func, callback=callback, check_output=check_output, keep=keep)


def main():
    class Test:
        i = 0

        @func_state(cb=lambda x: print("x", x))
        def x(self, v: int) -> int:
            p = self.i
            self.i += v
            return p

        @func_state(cb=lambda x: print("y", x))
        def y(self, v: int) -> int:
            p = self.i
            self.i *= v
            return p + v

    test = Test()
    test.x(2)
    test.y(2)
    test.x(2)
    test.y(2)
    test.x.refresh()
    test.y.refresh()
    test.x(4)
    test.y(4)
    test.x(4)
    test.y(4)

    @func_state(cb=lambda x: print("task", x))
    def task(v: int) -> int:
        test.x.refresh()
        test.x(v)
        return test.i + 3

    task(2)
    task(2)
    task.refresh()
    task(2)

    other = Test()
    # other.x.refresh()
    # other.y.refresh()
    print("new instance")
    print("other", other.i)
    task(2)
    task.refresh()
    task(2)
    print("other", other.i)
    other.x(2)
    other.x(2)
    print("other", other.i)
    other.y(2)
    other.y(2)
    print("other", other.i)


if str(__name__).upper() in ("__MAIN__",):
    main()

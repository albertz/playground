#!/usr/bin/env python3

"""
Idea:
Use `with` (context manager) as `if` condition.

Like this:
```
with IfConditionContextManager(True):
    # This code will be executed...

with IfConditionContextManager(False):
    # This code will not be executed...
```

Idea for implementation: Raise exception in `__enter__`.
Probably not possible, as this would not be caught by `__exit__`.

Conclusion:
Not possible.

"""


def _async_raise(target_tid: int, exception_class):
    if target_tid < 0:
        import _thread
        target_tid = _thread.get_ident()
    # Also see: https://stackoverflow.com/questions/36484151/throw-an-exception-into-another-thread
    import ctypes
    ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), ctypes.py_object(exception_class))
    # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    if ret == 0:
        raise ValueError("Invalid thread ID")
    elif ret > 1:
        # Huh? Why would we notify more than one threads?
        # Because we punch a hole into C level interpreter.
        # So it is better to clean up the mess.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class IfConditionContextManager:
    def __init__(self, condition: bool):
        self.condition = condition

    def __xxenter__(self):
        # raise Exception("xx")
        # _async_raise(-1, Exception)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exit:", exc_type, exc_val)

    def __getattribute__(self, item):
        print("__getattribute__", item)
        return super(IfConditionContextManager, self).__getattribute__(item)


def main():

    import better_exchook
    better_exchook.install()

    with IfConditionContextManager(False):
        print("Should not get here.")


if __name__ == '__main__':
    main()

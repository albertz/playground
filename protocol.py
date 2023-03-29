"""
protocol testing
"""

from __future__ import annotations
from typing import Optional


try:
    from typing import Protocol
except ImportError:
    try:
        from typing_extensions import Protocol
    except ImportError:
        Protocol = object


class GetModelFunc(Protocol):
    """get model func"""

    def __call__(self, *, epoch: int, step: int) -> bool:
        ...


def func(*, epoch: int, step: int) -> bool:
    return step > epoch


def meta_func(adder: Optional[GetModelFunc] = None):
    assert adder
    print(adder(epoch=1, step=2))


meta_func(func)

"""
protocol testing

https://youtrack.jetbrains.com/issue/PY-59869/Type-check-with-Protocol-does-not-work-conditional-Protocol-import-defined-in-separate-file
"""

from __future__ import annotations
from protocol_get_model_func import GetModelFunc


def meta_func(func: GetModelFunc):
    print(func(epoch=1, step=0))


def main():
    def _func(*, epoch: int, step: int) -> bool:
        return epoch < step

    meta_func(_func)

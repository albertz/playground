
# https://youtrack.jetbrains.com/issue/PY-59868/Type-inference-wrong-after-conditional-assignment

from __future__ import annotations
from typing import Optional, Union, Callable


def func(f: Optional[Union[str, Callable[[], int]]] = None):
    if f is None:
        f = "train"
    if callable(f):
        f()
    elif f == "train":
        print("train")
    else:
        raise ValueError(f"invalid f {f!r}")

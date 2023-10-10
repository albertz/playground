
"""
https://youtrack.jetbrains.com/issue/PY-63497/Incorrect-type-inference-to-None
"""

from typing import Optional


class Foo:
    def __init__(self):
        self.attr: Optional[str] = "hello"


class Bar:
    def __init__(self):
        self.foo = Foo()


def other():
    return False


def func(bar: Bar) -> str:
    if bar.foo and bar.foo.attr is not None and other():
        return bar.foo.attr

    if bar.foo and bar.foo.attr:
        # Type inference is wrong: PyCharm infers that bar.foo.attr is None here.
        # It prints the wrong warning: Cannot find reference 'upper' in 'None'
        return bar.foo.attr.upper()

    return "?"


print(func(Bar()))

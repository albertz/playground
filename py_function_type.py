
"""
https://youtrack.jetbrains.com/issue/PY-45382
"""


import types

def f():
  pass


# warning: Expected type 'Union[type, tuple[Union[type, tuple[Any, ...]], ...]]', got 'MethodType' instead
assert isinstance(f, types.FunctionType)

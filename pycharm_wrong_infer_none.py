
"""
https://youtrack.jetbrains.com/issue/PY-50411
"""


from typing import Optional, Tuple


class Obj:
  def __init__(self, a: Optional[int]):
    self.a = a


def func(obj: Obj):
  # No warning here.
  print(None if obj.a is None else (obj.a + 1))


def func_loop(objs: Tuple[Obj]):
  for obj in objs:
    # Warning on the second `obj.a`: Expected type 'int', got 'None' instead
    print(None if obj.a is None else (obj.a + 1))

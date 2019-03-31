
import typing


class X:
  def __init__(self):
    self.num = 42


class Y:
  def __init__(self):
    X_ = X
    self.x = None  # type: typing.Optional[X_]


def f():
  """
  https://youtrack.jetbrains.com/issue/PY-35022
  """
  y = Y()
  x = y.x  # inferred type for `x` is `Any`

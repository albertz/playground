
import contextlib


@contextlib.contextmanager
def g():
  """
  Dummy ctx.
  """
  yield


_direction = -1


def f():
  """
  https://youtrack.jetbrains.com/issue/PY-35020
  """
  with g():
    if _direction == 2:
      raise ValueError("?")
    y = 42
  if _direction == -1:
    y = y + 1
  return y  # warning: Local variable 'y' might be referenced before assignment

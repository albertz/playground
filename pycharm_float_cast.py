
import sys

PY3 = sys.version_info[0] >= 3

if PY3:
  import builtins
  unicode = str
else:
  # noinspection PyUnresolvedReferences
  import __builtin__ as builtins
  unicode = builtins.unicode


def g():
  """
  :rtype: object
  """
  return "42.0"


def f():
  """
  https://youtrack.jetbrains.com/issue/PY-35019
  """
  value = g()
  if value is not None:
    if isinstance(value, (str, unicode)):
      value = float(value)  # warning: expected type ..., got 'object' instead
    assert isinstance(value, (int, float))

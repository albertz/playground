
"""
https://youtrack.jetbrains.com/issue/PY-43913
"""

import typing


def test(obj):
  """
  :param typing.Any obj:
  """
  if hasattr(obj, "__getitem__"):
    elem0 = obj.__getitem__(0)
    print(elem0)

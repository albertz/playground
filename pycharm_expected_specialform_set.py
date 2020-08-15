
"""
https://youtrack.jetbrains.com/issue/PY-44052
"""

import typing


def get(x=None):
  """
  :param typing.Optional[typing.Set[int]] x:
  """


# warning: Expected type '_SpecialForm[Set[int]]', got 'Set[int]' instead
get(x={0})


"""
https://youtrack.jetbrains.com/issue/PY-50414
"""


def func():
  a, b = 1, 2
  ls = list([1, 2, 3])
  # Warning: Class 'list' does not define '__getitem__', so the '[]' operator cannot be used on its instances
  ls[a], ls[b] = ls[b], ls[a]

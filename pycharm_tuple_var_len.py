
# https://youtrack.jetbrains.com/issue/PY-36969


def foo(cond):
  """
  :param bool cond:
  :rtype: int
  """
  xs = (1, 2)
  if cond:
    xs += (3,)
  if not cond:
    a, b = xs
    c = 0
  else:
    a, b, c = xs  # PyCharm warning: Need more values to unpack
  return a + b + c



def g():
  """
  Replacement for :func:`f`.
  """
  return None


def f():
  """
  https://youtrack.jetbrains.com/issue/PY-35013
  """
  global f  # warning: Global variable 'f' is undefined at the module level
  f = g

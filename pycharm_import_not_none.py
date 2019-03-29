

def f():
  """
  https://youtrack.jetbrains.com/issue/PY-34985
  """
  g = None
  try:
    from maybe_existing_mod import maybe_existing_func as g
  except ImportError:
    pass
  if g:
    g()  # warning: 'g' is not callable

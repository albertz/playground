
"""
https://youtrack.jetbrains.com/issue/PY-41225
"""


def test():
  try:
    import better_exchook
    better_exchook.install()
  except ImportError:
    pass

  print("Foo")

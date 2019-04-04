
class Foo:
  """Dummy"""


def f():
  """
  https://youtrack.jetbrains.com/issue/PY-35165
  """
  Bar = Foo  # warning: Variable in function should be lowercase
  return Bar()

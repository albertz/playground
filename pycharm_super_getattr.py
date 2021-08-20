
"""
https://youtrack.jetbrains.com/issue/PY-50410
"""


class Obj:
  def __getattr__(self, item):
    # Warning: Unresolved attribute reference '__getattr__' for class 'object'
    return super(Obj, self).__getattr__(item)

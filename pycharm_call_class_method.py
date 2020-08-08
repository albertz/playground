
"""
https://youtrack.jetbrains.com/issue/PY-43915
"""

import sys


class Foo:
  def method(self, arg1):
    print(self, arg1)


if sys.version_info[0] == 3:
  Bar = Foo
else:
  Bar = globals()["Foo"]  # in practice sth else, just for this demo ...


class_method = Bar.method


def new_method(foo, arg1):
  return class_method(foo, arg1=arg1)  # warning: Unexpected argument


Bar.method = new_method

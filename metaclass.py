
"""
https://youtrack.jetbrains.com/issue/PY-41380
https://stackoverflow.com/questions/61447394/metaclass-call-vs-baseclass-new
"""

import better_exchook
better_exchook.install()


class Metaclass(type):
  def __call__(cls, *args):
    print("Metaclass.__call__", cls, args)
    obj = type.__call__(cls, *args)
    return obj  # cls.__init__ was already called


class DemoWithMetaclass(metaclass=Metaclass):
  def __init__(self, x):
    print("DemoWithMetaclass.__init__", x)
    self.x = x


class Baseclass(object):
  def __new__(cls, *args):
    print("Baseclass.__new__", cls, args)
    obj = super(Baseclass, cls).__new__(cls)
    return obj  # cls.__init__ not yet called


class DemoWithBaseclass(Baseclass):
  def __init__(self, x):
    print("DemoWithBaseclass.__init__", x)
    self.x = x


demo1 = DemoWithMetaclass(1)
print("got:", demo1, demo1.x)

demo2 = DemoWithBaseclass(2)
print("got:", demo2, demo2.x)

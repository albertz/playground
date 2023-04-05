"""
Pickling a dynamically created class
"""


import pickle
import copyreg


class Metaclass(type):
    """
    __getstate__ and __reduce__ do not work.
    However, we can register this via copyreg. See below.
    """


class Base:
    """Some base class. Does not really matter, you could also use `object`."""


def create_cls(name):
    return Metaclass(name, (Base,), {})


cls = create_cls("MyCustomObj")
print(f"{cls=}")


def _reduce_metaclass(cls):
    metaclass = cls.__class__
    cls_vars = dict(vars(cls))
    cls_vars.pop("__dict__", None)
    cls_vars.pop("__weakref__", None)
    print("reduce metaclass", cls, metaclass, cls.__name__, cls.__bases__, vars(cls))
    return metaclass, (cls.__name__, cls.__bases__, cls_vars)


copyreg.pickle(Metaclass, _reduce_metaclass)


cls = pickle.loads(pickle.dumps(cls))
print(f"{cls=} after pickling")

a = cls()
print(f"instance {a=}, {a.__class__=}, {a.__class__.__mro__=}")
a = pickle.loads(pickle.dumps(a))
print(f"instance {a=} after pickling, {a.__class__=}, {a.__class__.__mro__=}")

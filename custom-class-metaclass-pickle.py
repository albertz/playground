"""
Pickling a dynamically created class
"""


import pickle
import copyreg


class Metaclass(type):
    # __getstate__ and __reduce__ do not work
    pass


class Obj(metaclass=Metaclass):
    pass


def create_cls(name):
    return Metaclass(name, (Obj,), {})


cls = create_cls("MyCustomObj")
print(cls)


def _reduce_metaclass(cls):
    metaclass = cls.__class__
    cls_vars = dict(vars(cls))
    cls_vars.pop("__dict__", None)
    cls_vars.pop("__weakref__", None)
    print("reduce metaclass", cls, metaclass, cls.__name__, cls.__bases__, vars(cls))
    return metaclass, (cls.__name__, cls.__bases__, cls_vars)


copyreg.pickle(Metaclass, _reduce_metaclass)


cls = pickle.loads(pickle.dumps(cls))
print(cls)

a = cls()
print(a, a.__class__, a.__class__.__mro__)
a = pickle.loads(pickle.dumps(a))
print(a, a.__class__, a.__class__.__mro__)

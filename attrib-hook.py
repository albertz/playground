"""
You have some class with attribs,
and now you want to hook into any attrib read (maybe write) access.

The class is maybe derived from PyTorch nn.Module,
and the attrib is maybe a parameter.

Think about how to implement weight-norm,
but not using the forward-pre-hook,
but instead using such an attrib-hook.

"""


from typing import Any, Type, Dict
import pickle
import copyreg


class _AttrHookMixinDemo:
    def __getattr__(self, name):
        print("getattr", name)
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def __getattribute__(self, item):
        print("getattribute", item)
        return super().__getattribute__(item)


class Module:
    def __init__(self):
        self.param = "some-param"


m = Module()
print(m.param)
try:
    print(m.non_existing)
except AttributeError as exc:
    print("got expected AttributeError:", exc)


def _setup_mixin_class_in_instance(obj: Any, mixin_cls: Type):
    obj.__class__ = type(f"{obj.__class__.__name__}({mixin_cls.__name__})", (mixin_cls, obj.__class__), {})


def _remove_mixin_class_in_instance(obj: Any, mixin_cls: Type):
    assert len(obj.__class__.__bases__) == 2 and obj.__class__.__bases__[0] is mixin_cls
    obj.__class__ = obj.__class__.__bases__[1]


print("*** setup mixin class demo ***")
_setup_mixin_class_in_instance(m, _AttrHookMixinDemo)
print(m)
print(m.param)
print(m.non_existing)  # no error, gets None


print("*** remove mixin class demo ***")
_remove_mixin_class_in_instance(m, _AttrHookMixinDemo)
print(m)
print(m.param)


# ------


class _AttrHookMixin:
    _attr_hooks: Dict[str, Any]  # attrib name -> descriptor

    def __getattribute__(self, item):
        if item == "_attr_hooks":
            try:
                return super().__getattribute__(item)
            except AttributeError:
                return {}
        print("getattribute", item)
        attr_hooks = self._attr_hooks
        if item in attr_hooks:
            return attr_hooks[item].get(self, item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        print("setattr", key)
        if key in self._attr_hooks:
            return self._attr_hooks[key].set(self, value)
        return super().__setattr__(key, value)


print("*** setup mixin class ***")
_setup_mixin_class_in_instance(m, _AttrHookMixin)
print(m)
m._attr_hooks = {}
print(m.param)
try:
    print(m.non_existing)
except AttributeError as exc:
    print("got expected AttributeError:", exc)
m.param2 = "some-param2"
print(m.param2)


class ParamHookDescriptor:
    # It's somewhat like a Python descriptor (https://docs.python.org/3/howto/descriptor.html)
    # but not exactly, the __get__/__set__ API is different.
    def get(self, obj, name):
        print("get param")
        return obj.__dict__[name] * 2


m._attr_hooks["param"] = ParamHookDescriptor()
print(m.param)
# m.param = "some-param-new"  # not allowed because descriptor has no __set__
_remove_mixin_class_in_instance(m, _AttrHookMixin)


# Some other alternative:


class PickleableType(type):
    pass


def _setup_hooks(obj, hooks: Dict[str, property]):
    hooks_cls_name_postfix = f"(with hooks)"
    if obj.__class__.__name__.endswith(hooks_cls_name_postfix):
        hooks_cls = obj.__class__
    else:
        hooks_cls = PickleableType(f"{obj.__class__.__name__}{hooks_cls_name_postfix}", (obj.__class__,), {})
        obj.__class__ = hooks_cls
    for name, hook in hooks.items():
        setattr(hooks_cls, name, hook)


def _get_param(self: Module):
    print("get param")
    return self.__dict__["param"] * 3


print("*** setup attr hooks via property ***")
_setup_hooks(m, {"param": property(_get_param)})
print(m)
print(m.param)


def _reduce_metaclass(cls):
    metaclass = cls.__class__
    cls_vars = dict(vars(cls))
    cls_vars.pop("__dict__", None)
    cls_vars.pop("__weakref__", None)
    print("reduce metaclass", cls, metaclass, cls.__name__, cls.__bases__, vars(cls))
    return metaclass, (cls.__name__, cls.__bases__, cls_vars)


# pickling such module is problematic because of the dynamically created type.
# however, with the metaclass, and copyreg, it works.
copyreg.pickle(PickleableType, _reduce_metaclass)


def _reduce_property(prop):
    print("reduce property", prop)
    return property, (prop.fget, prop.fset, prop.fdel, prop.__doc__)


copyreg.pickle(property, _reduce_property)

# now this works:
m = pickle.loads(pickle.dumps(m))


print(m)
print(m.param)

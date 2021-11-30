
import better_exchook
better_exchook.install()


class Metaclass(type):
  def __call__(cls, *args):
    print("Metaclass.__call__", cls, args)
    obj = type.__call__(cls, *args)
    return obj  # cls.__init__ was already called

  def __instancecheck__(self, instance):
    print("Metaclass.__instancecheck__", instance)
    if isinstance(instance, tuple):
      return True
    return super(Metaclass, self).__instancecheck__(instance)

  def __subclasscheck__(self, subclass):
    print("Metaclass.__subclasscheck__", subclass)
    if subclass == tuple:
      return True
    return super(Metaclass, self).__instancecheck__(subclass)


class DemoWithMetaclass(metaclass=Metaclass):
  def __init__(self):
    print("DemoWithMetaclass.__init__")


x = DemoWithMetaclass()

print("tuple:", isinstance(x, tuple))
print("tuple:", issubclass(type(x), tuple))
print("tuple:", isinstance((), DemoWithMetaclass))
print("tuple:", issubclass(tuple, DemoWithMetaclass))

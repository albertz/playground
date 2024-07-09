"""
Playing around with descriptors and properties.

https://docs.python.org/3/howto/descriptor.html
"""


def f(*args):
    print("f", args)
    return "f-return"


class A:
    # property creates a property object, with PyProperty_Type (descrobject.c),
    # which has tp_descr_get = property_descr_get, which basically just calls the
    # given fget function, in this case f.
    x = property(f)


a = A()
print("a.x:", a.x)


# This cannot work: Descriptors are only searched in the class, not in the instance.
a.y = property(f)
print("a.y:", a.y)

# ---


class Property(property):
    def __init__(self, name, desc, **kwargs):
        property.__init__(self, **kwargs)
        self.name = name
        self.desc = desc

    def __str__(self):
        return "Property %s: %s" % (self.name, self.desc)


class B:
    x = Property("x", "test", fget=f)


b = B()
print("b.x:", b.x)
print(b.__class__.x)


# ---


class InitBy(property):
    def __init__(self, init_func):
        super().__init__(fget=self.fget)
        self.init_func = init_func

    def fget(self, inst):
        if hasattr(self, "value"):
            return self.value
        self.value = self.init_func()
        return self.value


def init_func_test():
    print("init_func_test")
    return "x"


class C:
    x = InitBy(init_func_test)


c = C()
print("c.x:", c.x)
print("c.x:", c.x)

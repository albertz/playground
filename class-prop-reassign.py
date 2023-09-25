class Foo:
    __slots__ = ('_x',)

    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        print("x property getter")
        return self._x


x = Foo(42)
assert x.x == 42


Foo.x = 43

assert x.x == 43


def y_property_getter(self):
    print("y property getter")
    return self._x - 1


Foo.x = property(y_property_getter)

assert x.x == 41

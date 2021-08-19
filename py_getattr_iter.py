

class Dyn:
  def __init__(self):
    self.dummy_existing_attrib = 42

  def __getattr__(self, item):
    print(f"{self}.__getattr__({item!r})")
    if item == "__iter__":
      return _custom_iter()
    return super(Dyn, self).__getattr__(item)

  def __getattribute__(self, item):
    print(f"{self}.__getattribute__({item!r})")
    return super(Dyn, self).__getattribute__(item)

  def __repr__(self):
    return "Dyn()"


def _custom_iter():
  return iter([1, 2, 3])


dyn = Dyn()
hasattr(dyn, "dummy_existing_attrib")
hasattr(dyn, "test_hasattr")
# Output so far:
# Dyn().__getattribute__('dummy_existing_attrib')
# Dyn().__getattribute__('test_hasattr')
# Dyn().__getattr__('test_hasattr')
# So, all "normal" attrib access gets tracked.
# __getattribute__ catches all getattr/hasattr,
# __getattr__ is called for non-existing.
# "Special" attrib access, e.g. for __repr__ etc,
# is *not* tracked!
# So that is why this does not work:
# a, b, c = dyn
# TypeError: cannot unpack non-iterable Dyn object
# It also doesn't work if we assign __iter__ like:
# dyn.__iter__ = _custom_iter


# Just for demonstration, it works when you explicitly define __iter__ in the class.

class IterObj:
  def __iter__(self):
    print("iter")
    return _custom_iter()


iter_obj = IterObj()
a, b, c = iter_obj
print(a, b, c)


# So, solution to handle __iter__ and co dynamically? Need to copy the class.

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


# ---

class Property(property):
	def __init__(self, name, desc, **kwargs):
		property.__init__(self, **kwargs)
		self.name = name
		self.desc = desc
	def __str__(self): return "Property %s: %s" % (self.name, self.desc)
	
class B:
	x = Property("x", "test", fget = f)

b = B()
print("b.x:", b.x)
print(b.__class__.x)


# ---

class initBy(property):
	def __init__(self, initFunc):
		property.__init__(self, fget = self.fget)
		self.initFunc = initFunc
	def fget(self, inst):
		if hasattr(self, "value"): return self.value
		self.value = self.initFunc()
		return self.value

def initFuncTest():
	print("initFuncTest")
	return "x"
	
class C:
	x = initBy(initFuncTest)

c = C()
print("c.x:", c.x)
print("c.x:", c.x)

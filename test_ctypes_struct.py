from ctypes import *

def WrapClass(c):
	class C(c): pass
	C.__name__ = "wrapped_" + c.__name__
	return C

class S(Structure):
    _fields_ = [("x", POINTER(c_int)), ("y", WrapClass(c_int))]

o = S()
print o.x
print o.y
print o.y.value

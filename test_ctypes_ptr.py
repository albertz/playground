import ctypes, _ctypes

def castArgToCtype(arg, ctyp):
    if issubclass(ctyp, _ctypes._Pointer): return createPtr(arg, ctyp)
    return ctyp(arg)

def createPtr(arg, ctyp):
    assert isinstance(arg, (list,tuple))
    assert issubclass(ctyp, _ctypes._Pointer)
    o = (ctyp._type_ * (len(arg) + 1))()
    for i in xrange(len(arg)):
        o[i] = castArgToCtype(arg[i], ctyp._type_)
    op = ctypes.pointer(o)
    op = ctypes.cast(op, ctyp)
    return op

a = createPtr((1,2,3), ctypes.POINTER(ctypes.c_int))
print a, a[0], a[1], a[2], a._objects


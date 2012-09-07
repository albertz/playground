# test this module by:
#   python -c "import test_importearlyexit"
# the wanted output is just 'mod: start'

print "mod: start"

if True: # check if we should skip module loading here
	import inspect
	frame = inspect.currentframe()
	code = frame.f_code
	
	import ctypes
	ctypes.pythonapi.PyString_AsString.argtypes = (ctypes.c_void_p,)
	ctypes.pythonapi.PyString_AsString.restype = ctypes.POINTER(ctypes.c_char)
	
	import dis
	
	while True:
		targetjumpptr = len(code.co_code) - 4 # LOAD_CONST+RETURN_VALUE at end
		iptr = frame.f_lasti
		
		codestr = ctypes.pythonapi.PyString_AsString(id(code.co_code))
		codestr[iptr] = chr(dis.opmap["JUMP_ABSOLUTE"])
		codestr[iptr+1] = chr(targetjumpptr & 255)
		codestr[iptr+2] = chr(targetjumpptr >> 8)

print "mod: after early exit"

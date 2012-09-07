# test this module by:
#   python -c "import test_importearlyexit"
# the wanted output is just 'mod: start'

print "mod: start"

def breakModule():
	import inspect
	frame = inspect.currentframe()
	frame = frame.f_back # manipulate the caller frame
	code = frame.f_code
	
	import ctypes
	ctypes.pythonapi.PyString_AsString.argtypes = (ctypes.c_void_p,)
	ctypes.pythonapi.PyString_AsString.restype = ctypes.POINTER(ctypes.c_char)
	
	import dis
	
	targetjumpptr = len(code.co_code) - 4 # LOAD_CONST+RETURN_VALUE at end
	iptr = frame.f_lasti + 3 # this is a funccall, so advance by 3
	
	codestr = ctypes.pythonapi.PyString_AsString(id(code.co_code))
	codestr[iptr] = chr(dis.opmap["JUMP_ABSOLUTE"])
	codestr[iptr+1] = chr(targetjumpptr & 255)
	codestr[iptr+2] = chr(targetjumpptr >> 8)
	

if True: # check if we should skip module loading here
	breakModule()
	
print "mod: after early exit"

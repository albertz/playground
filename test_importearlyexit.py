# test this module by:
#   python -c "import test_importearlyexit"
# the wanted output is just 'mod: start'

print "mod: start"

def breakModule():
	"""
	Be careful using this! It manipulates the code string of the code object
	of the caller frame! It will overwrite the next 3 bytecodes.
	
	There are a few expectations that this work:
	 - inspect.currentframe().f_back must return the caller frame
	 - id(code.co_code) must return the native address of the code string
	 - the C-API PyString_AsString must be available and callable via ctypes
	
	Discussion: The bytecode for Python modules allows the op RETURN_VALUE.
	This is what you see at the compiled bytecode at the end of a module.
	I wonder why `return` is not allowed at the module source level.
	A normal `return` in a module should just work.
	"""
	import inspect
	frame = inspect.currentframe()
	frame = frame.f_back # manipulate the caller frame
	code = frame.f_code
	
	import ctypes
	ctypes.pythonapi.PyString_AsString.argtypes = (ctypes.c_void_p,)
	ctypes.pythonapi.PyString_AsString.restype = ctypes.POINTER(ctypes.c_char)
		
	targetjumpptr = len(code.co_code) - 4 # LOAD_CONST+RETURN_VALUE at end
	iptr = frame.f_lasti + 3 # this is a funccall, so advance by 3
	assert iptr + 3 <= targetjumpptr # otherwise we don't have enough room
	
	import dis
	codestr = ctypes.pythonapi.PyString_AsString(id(code.co_code))
	codestr[iptr] = chr(dis.opmap["JUMP_ABSOLUTE"])
	codestr[iptr+1] = chr(targetjumpptr & 255)
	codestr[iptr+2] = chr(targetjumpptr >> 8)
	

if True: # check if we should skip module loading here
	breakModule()
	
print "mod: after early exit"

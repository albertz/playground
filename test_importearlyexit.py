# test this module by:
#   python -c "import test_importearlyexit"
# the wanted output is just 'mod: start'

from __future__ import print_function
import sys

Py2 = sys.version_info[0] == 2

print("mod: start")


def break_module():
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
  frame = frame.f_back  # manipulate the caller frame
  code = frame.f_code

  import ctypes
  if Py2:
    c_bytes_as_str = ctypes.pythonapi.PyString_AsString
  else:
    c_bytes_as_str = ctypes.pythonapi.PyBytes_AsString  # python 3
  c_bytes_as_str.argtypes = (ctypes.c_void_p,)
  c_bytes_as_str.restype = ctypes.POINTER(ctypes.c_char)

  targetjumpptr = len(code.co_code) - 4  # LOAD_CONST+RETURN_VALUE at end
  iptr = frame.f_lasti + (3 if Py2 else 4)  # this is a funccall, so advance by 4
  assert iptr + (3 if Py2 else 4) <= targetjumpptr  # otherwise we don't have enough room

  import dis
  codestr = c_bytes_as_str(id(code.co_code))
  # dis.dis(ctypes.string_at(codestr, len(code.co_code)))  # for debugging
  codestr[iptr] = dis.opmap["JUMP_ABSOLUTE"]
  codestr[iptr+1] = targetjumpptr & 255
  codestr[iptr+2] = targetjumpptr >> 8


if True:  # check if we should skip module loading here
  break_module()

print("mod: after early exit")

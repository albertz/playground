# test this module by:
#   python -c "import test_importearlyexit"
# the wanted output is just 'mod: start'

print "mod: start"

import dis
cstr = []
#cstr += [dis.opmap["POP_BLOCK"]]
#cstr += [dis.opmap["JUMP_ABSOLUTE"],255,255]
cstr += [dis.opmap["LOAD_CONST"], 0, 0]
cstr += [dis.opmap["RETURN_VALUE"]]

#flags = 0
flags = 1 + 2 + 64 # that is the default if it is a local function

import types
c = types.CodeType(
	0, # argcount
	0, # nlocals
	1, # stacksize
	flags, # flags
	"".join(map(chr, cstr)), # codestring
	(None,), # constants
	(), # names
	(), # varnames
	"<code>", # filename
	"exit", # name
	1, # firstlineno
	"", # lnotab
	)


exec c

print "mod: after early exit"

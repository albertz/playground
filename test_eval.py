from ast import *
#import ast

class D:
	def __init__(self):
		self.d = {}		
	def __getitem__(self, k):
		print "D get", k
		return self.d[k]
	def __setitem__(self, k, v):
		print "D set", k, v
		self.d[k] = v
	def __getattr__(self, k):
		print "D attr", k
		raise AttributeError
	
globalsDict = D()

exprAst = Interactive(body=[
	FunctionDef(
		name='foo',
		args=arguments(args=[], vararg=None, kwarg=None, defaults=[]),
		body=[Pass()],
		decorator_list=[])])

fix_missing_locations(exprAst)

src = "def foo(): print x"

compiled = compile(src, "<foo>", "exec")
exec compiled in {}, globalsDict
#exec src in {}, globalsDict

#import types
#f = types.FunctionType(code=compiled, globals={})

f = globalsDict["foo"]
print(f)

f()

#x = 44
#g = f()

#print g
#g = types
#g()

def testfunc0(): print "testfunc0"
def testfunc1(a1): print "testfunc1", a1
def testfunc2(a1,a2): print "testfunc2", a1, a2

class A:
	def func0(): print "func0"
	def func1(a1): print "func1", a1
	def func2(a1,a2): print "func2", a1, a2
	varfunc0 = testfunc0
	varfunc1 = testfunc1
	varfunc2 = testfunc2
	var = 42

	def __init__(self):
		self.initvarfunc0 = testfunc0
		self.initvarfunc1 = testfunc1
		self.initvarfunc2 = testfunc2
		self.initvar = 42.5

# A.func0() # doesnt work
# A.func1("arg1") # doesnt work
# A.func2("arg1","arg2") # doesnt work
# A.varfunc0() # doesnt work
# A.varfunc1("arg1") # doesnt work
# A.varfunc2("arg1","arg2") # doesnt work
print "A.func0 =", A.func0 # this works though!
print "A.varfunc0 =", A.varfunc0 # this too
print "A.var =", A.var

a = A()
# a.func0() # this doesnt work of course - there must be at least one arg
a.func1()
a.func2("arg2")
# a.varfunc0() # doesnt work for same reason as a.func0
a.varfunc1()
a.varfunc2("arg2")

a.initvarfunc0()
a.initvarfunc1("arg1")
a.initvarfunc2("arg1","arg2")
print "a.initvar =", a.initvar

A.anothertest = testfunc1
a.anothertest()
# A.anothertest("arg1") # doesnt work
print "testfunc1 =", testfunc1
print "A.anothertest =", A.anothertest
print "a.anothertest =", a.anothertest

A.anothertest = 43
print "A.anothertest =", A.anothertest
print "a.anothertest =", a.anothertest

setattr(A, "anothertest", testfunc1)
a.anothertest()
print "A.anothertest = ", A.anothertest
print "A.anothertest getattr ", getattr(A, "anothertest")

setattr(a, "anothertest", testfunc1)
a.anothertest("arg1")
print "a.anothertest =", a.anothertest
print "a.anothertest getattr", getattr(a, "anothertest")
delattr(a, "anothertest")

print "a.func1 =", a.func1
f = a.func1
print "f =", f
A.anothertest = f
print "A.anothertest =", A.anothertest
print "a.anothertest =", a.anothertest
A.anothertest()
a.anothertest()

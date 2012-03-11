#!/usr/bin/python

class Foo:
	def __init__(self): self.x = 42
	def bla(fooself):
		class Bar:
			def bla(barself):
				print "Bar.bla, x =", fooself.x
				fooself.x += 1
		return Bar()
	
f = Foo()
f.bla().bla()
f.bla().bla()

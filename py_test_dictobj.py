#!/usr/bin/python

class Foo1(dict):
	def __getattr__(self, key): return self[key]
	def __setattr__(self, key, value): self[key] = value

class Foo2(dict):
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__

o1 = Foo1()
o1.x = 42
print(o1, o1.x)

o2 = Foo2()
o2.x = 42
print(o2, o2.x)

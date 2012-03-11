#!/usr/bin/python

import gc
#gc.set_debug(gc.DEBUG_LEAK)

class Foo(dict):
	# this leaks:
	#def __init__(self):
	#	self.__dict__ = self

	# this is ok:
   def __getattr__(self, key):
      return self[key]
   
   def __setattr__(self, key, value):
      self[key] = value

while True:
	o = Foo()
	gc.collect()


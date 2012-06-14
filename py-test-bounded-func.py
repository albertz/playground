#!/usr/bin/python

# discussion: http://stackoverflow.com/questions/11036815/implicitely-bound-callable-objects-to-instance

import itertools

def bound(f):
	def dummy(*args, **kwargs):
		return f(*args, **kwargs)
	return dummy

class LFSeq: # lazy infinite sequence with new elements from func
	def __init__(self, func):
		self.evaluated = []
		self.func = func
	@bound
	class __iter__:
		def __init__(self, seq):
			self.index = 0
			self.seq = seq
		def next(self):
			if self.index >= len(self.seq.evaluated):
				self.seq.evaluated += [self.seq.func()]
			self.index += 1
			return self.seq.evaluated[self.index - 1]

def test():
	f = itertools.count(0).next	
	s = LFSeq(f)
	print s.__iter__
	# LFSeq.__iter__(s) == s.__iter__() ?
	i = iter(s)
	#i = s.__iter__()
	#i = LFSeq.__iter__(s)
	print i
	for c in range(10):
		print next(i)
		
if __name__ == '__main__':
	test()

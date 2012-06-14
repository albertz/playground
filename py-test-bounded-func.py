#!/usr/bin/python

# discussion: http://stackoverflow.com/questions/11036815/implicitely-bound-callable-objects-to-instance

import itertools
import random

def bound(f):
	def dummy(*args, **kwargs):
		return f(*args, **kwargs)
	return dummy

class LFSeq: # lazy infinite sequence with new elements from func
	def __init__(self, func):
		self.evaluated = []
		self.func = func

	def fillUpToLen(self, n):
		self.evaluated += [self.func() for i in range(len(self.evaluated), n)]
		
	# nicer, as suggested on SO, would be a generator because Python bounds that correctly
	# this is just for demonstration, though...
	@bound
	class __iter__:
		def __init__(self, seq):
			self.index = 0
			self.seq = seq
		def next(self):
			self.index += 1
			return self.seq[self.index]

	def __getitem__(self, i):
		self.fillUpToLen(i + 1)
		return self.evaluated[i]

	def __getslice__(self, i, k):
		assert k is not None, "inf not supported here"
		if i is None: i = 0
		assert i >= 0, "LFSeq has no len"
		assert k >= 0, "LFSeq has no len"
		self.fillUpToLen(k)
		return self.evaluated[i:k]
		
LRndSeq = lambda: LFSeq(lambda: chr(random.randint(0,255)))

class LList: # lazy list
	def __init__(self, base, op=iter):
		self.base = base
		self.op = op
	def __add__(self, other):
		return LList((self, other), lambda x: itertools.chain(*x))
	def __iter__(self):
		return self.op(self.base)
	def __str__(self):
		return "llist(%s,%s)" % (self.base, self.op)
	def __getslice__(self, start, end):
		# slow dummy implementation
		if start is None: start = 0
		tmp = None
		for i, v in itertools.izip(itertools.count(0), iter(self)):
			if i >= end: break
			if i == start: tmp = v
			if i > start: tmp += v
		return tmp

def commonStrLen(*args):
	c = 0
	for cs in itertools.izip(*args):
		if min(cs) != max(cs): break
		c += 1
	return c
	
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
		
def test2():
	s = LRndSeq()
	print repr("".join(s[:10]))
	
def test3():
	from sha import sha
	key = "foo"
	longDevId = LList("dev-" + sha(key).hexdigest()) + LRndSeq()
	print longDevId
	print repr(longDevId[:100])
	print repr(longDevId[:100])

def test4():
	from sha import sha
	key = "foo"
	longDevId1 = LList("dev-" + sha(key).hexdigest()) + "-" + LRndSeq()
	longDevId2 = "dev-" + sha(key).hexdigest() + "-bla"
	print commonStrLen(longDevId1, longDevId2)
	
if __name__ == '__main__':
	test()
	test2()
	test3()
	test4()
	
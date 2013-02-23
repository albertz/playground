
import testcrash_python_threadlocal
Test = testcrash_python_threadlocal.Test

EnableWorkaround = False

import sys, os, time, signal
from threading import Thread, local, RLock
from weakref import ref

class _L:	
	def __init__(self):
		self.l = None
		self.Test = Test
	
		if EnableWorkaround:
			self.refs = set()
			self.lock = RLock()
			class Guard:
				def __init__(gself):
					with self.lock:
						gself.test = Test()
						self.refs.add(ref(gself))
						assert ref(gself) in self.refs
				def __del__(gself):
					with self.lock:
						gself.test = None
						self.refs.discard(ref(gself))
			self.Test = Guard
	
	@property
	def test(self):
		if not self.l: self.l = local()
		if not getattr(self.l, "test", None):
			self.l.test = self.Test()
		return self.l.test
	
	def reset(self):
		print "reset now"
		self.l = None
		
		if EnableWorkaround:
			with self.lock:
				for g in self.refs:
					g = g()
					if not g: continue
					g.test = None

L = _L()

def setLocal(): getattr(L, "test")

def dummy(w):
	setLocal()
	time.sleep(w)

def dummy2(w):
	time.sleep(w)
	setLocal()
	while True:
		time.sleep(0.1)

def startThread(f):
	t = Thread(target=f)
	t.daemon = True
	t.start()

setLocal()
startThread(lambda: dummy(0.5))
startThread(lambda: dummy(2))
startThread(lambda: dummy2(0))

time.sleep(1)
L.reset()


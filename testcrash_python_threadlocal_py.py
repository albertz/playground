
import testcrash_python_threadlocal
Test = testcrash_python_threadlocal.Test

EnableWorkaround = False

import sys, os, time, signal
from threading import Thread, local, RLock

class _L:	
	def __init__(self):
		self.l = None
		self.Test = Test
	
		if EnableWorkaround:
			self.lock = RLock()
			class Guard:
				def __init__(gself):
					with self.lock:
						gself.test = Test()
				def __del__(gself):
					with self.lock:
						gself.test = None
			self.Test = Guard
			
	@property
	def test(self):
		if not self.l: self.l = local()
		if not getattr(self.l, "test", None):
			self.l.test = self.Test()
		return self.l.test
	
L = _L()

def setLocal(): getattr(L, "test")

def dummy(w):
	setLocal()
	time.sleep(w)

def dummy2():
	setLocal()
	while True:
		time.sleep(0.1)

def startThread(f):
	t = Thread(target=f)
	t.daemon = True
	t.start()

startThread(lambda: dummy(0.1))
startThread(lambda: dummy(2))
startThread(dummy2)
time.sleep(0.001)

time.sleep(1)
L.l = None


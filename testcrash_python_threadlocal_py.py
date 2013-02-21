
try:
	import faulthandler
	faulthandler.enable(all_threads=True)
except ImportError:
	print "note: faulthandler module not available"

import testcrash_python_threadlocal
Test = testcrash_python_threadlocal.Test

import sys, os, time, signal
from threading import Thread, local

class _L:
	l = None
	
	@property
	def test(self):
		if not self.l: self.l = local()
		if not getattr(self.l, "test", None):
			self.l.test = Test()
		return self.l.test
	
L = _L()

def setLocal(): getattr(L, "test")

def dummy(c, w1, w2):
	for i in range(c):
		time.sleep(w1)
		setLocal()
	time.sleep(w2)

def startThread(f):
	t = Thread(target=f)
	t.daemon = True
	t.start()

startThread(lambda: dummy(100, 0.001, 0.5))
startThread(lambda: dummy(100, 0.001, 2))
startThread(lambda: dummy(10, 0.05, 1))

time.sleep(1)
def remove():
	setLocal()
	L.l = None
startThread(remove)

# fun
os.kill(0, signal.SIGINT)

time.sleep(3)
sys.exit()

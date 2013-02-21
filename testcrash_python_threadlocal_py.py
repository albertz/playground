
try:
	import faulthandler
	faulthandler.enable(all_threads=True)
except ImportError:
	print "note: faulthandler module not available"

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

def dummy(c, w1, w2):
	setLocal()
	for i in range(c):
		time.sleep(w1)
		setLocal()
	time.sleep(w2)

def dummy2():
	setLocal()
	while True:
		time.sleep(0.1)

def startThread(f):
	t = Thread(target=f)
	t.daemon = True
	t.start()

for i in range(1):
	#startThread(lambda: dummy(100, 0.001, 0.5))
	#startThread(lambda: dummy(100, 0.001, 0))
	#startThread(lambda: dummy(100, 0.001, 1))
	#startThread(lambda: dummy(100, 0.00001, 0))
	#startThread(lambda: dummy(10, 0.05, 1))
	startThread(lambda: dummy(0, 0, 0.1))
	startThread(lambda: dummy(0, 0, 2))
	startThread(dummy2)
	time.sleep(0.001)

time.sleep(1)
def remove():
	#setLocal()
	time.sleep(0.5)
	L.l = None
startThread(remove)

# fun
#try: os.kill(0, signal.SIGINT)
#except KeyboardInterrupt: pass

time.sleep(3)
sys.exit()

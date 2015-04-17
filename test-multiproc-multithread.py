#!/usr/bin/env python2

import multiprocessing
from multiprocessing import Process
import threading
from threading import Thread
import thread
import os
import sys
import time
import signal


# Demo of: http://bugs.python.org/issue17094
# sys._current_frames() reports too many/wrong stack frame
# Also: http://stackoverflow.com/questions/8110920


def initBetterExchook():
  import thread
  import threading
  import better_exchook
  main_thread_id = thread.get_ident()

  def excepthook(exc_type, exc_obj, exc_tb):
    print "Unhandled exception %s in thread %s, proc %s." % (exc_type, threading.currentThread(), os.getpid())
    better_exchook.better_exchook(exc_type, exc_obj, exc_tb)

    if main_thread_id == thread.get_ident():
      print "We are the main thread."
      if not isinstance(exc_type, Exception):
        # We are the main thread and we got an exit-exception. This is likely fatal.
        # This usually means an exit. (We ignore non-daemon threads and procs here.)
        # Print the stack of all other threads.
        if hasattr(sys, "_current_frames"):
          print ""
          threads = {t.ident: t for t in threading.enumerate()}
          for tid, stack in sys._current_frames().items():
            if tid != main_thread_id:
              print "Thread %s:" % threads.get(tid, "unnamed with id %i" % tid)
              better_exchook.print_traceback(stack)
              print ""

  sys.excepthook = excepthook


def thread1():
	while True:
		pass

def thread2():
	while True:
		pass

def proc():
	print "Child proc, my threads:", threading.enumerate()
	try:
		while True:
			pass
	except BaseException:
		sys.excepthook(*sys.exc_info())

def main():
	print "Main proc, pid: %i" % os.getpid()

	initBetterExchook()

	t1 = Thread(name="T1", target=thread1)
	t1.daemon = True
	t1.start()
	t2 = Thread(name="T2", target=thread2)
	t2.daemon = True
	t2.start()

	p = Process(target=proc)
	p.daemon = True
	p.start()
	print "Child started, pid: %i" % p.ident

	time.sleep(1)
	os.kill(p.ident, signal.SIGINT)
	p.join()


if __name__ == "__main__":
	main()


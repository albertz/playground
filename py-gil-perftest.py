#!env python

# by David Beazley, 2010, in his talk about the GIL
#   http://www.dabeaz.com/GIL/
#   http://blip.tv/carlfk/mindblowing-python-gil-2243379

def count(n):
	while n > 0:
		n -= 1

N = 100 * 1000 * 1000

def serial():
	count(N)
	count(N)

def threaded():
	from threading import Thread
	t1 = Thread(target=count,args=(N,))
	t1.start()
	t2 = Thread(target=count,args=(N,))
	t2.start()
	t1.join()
	t2.join()

def timefunc(funcname):
	import timeit
	r = timeit.timeit(
		funcname + "()",
		setup="from __main__ import " + funcname,
		number=1)
	print(r)

timefunc("serial")
timefunc("threaded")




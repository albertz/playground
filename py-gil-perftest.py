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

# my results (MacBook Pro Early 2011, 2 GHz i7, 4 cores + HT):
#   CPython 2.7.5:
#     15.8614599705
#     23.1555030346
#   CPython 3.3.2:
#     19.368003961000795
#     20.278744241000823     
#   PyPy 2.0.0:
#     0.240994930267
#     0.622744083405     



#!/usr/bin/env python3

i = 0

def func():
	global i
	i += 1
	if i == 1: return
	if i == 2: raise Exception
	if i == 3: raise KeyboardInterrupt

def test():
	try:
		func()
	except Exception:
		pass
	except KeyboardInterrupt:
		return
	finally:
		print("Finally %i" % i)

test()
test()
test()


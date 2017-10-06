

def f(x):
	print("Hi from f(%r)" % x)
	return "assert failed: %r" % x


assert True, f(True)
assert False, f(False)


def f():
	x = 1
	def g():
		print "g x =", x
	g()
	x = 2
	g()
	
	def h():
		yield None
		print "h x =", x
		yield None
		print "h x =", x
	
	for _ in h():
		x += 1

f()

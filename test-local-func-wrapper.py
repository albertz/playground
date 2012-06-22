# http://stackoverflow.com/questions/11164930/strange-python-function-scope-behavior

class Bar:
	def __init__(self):
		for fn in ["open","openW","remove","mkdir","exists","isdir","listdir"]:
			print "register", fn
			def func_wrapper(filename):
				print "called func wrapper", fn, filename
			setattr(self, fn, func_wrapper)
			func_wrapper = None

bar = Bar()
bar.open("a")
bar.remove("b")
bar.listdir("c")


def p(s): print s

funcs = []
for i in range(5):
	#def test():
	#	print i
	test = (lambda n: (lambda: p(n)))(i)
	funcs += [test]

for f in funcs:
	f()


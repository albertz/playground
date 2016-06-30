
# http://stackoverflow.com/questions/38126674/python-how-to-access-outer-class-namespaces

import sys

class A:
	class B:
		a = 1
	class C:
		print("C locals: %r" % locals())
		print("C parent locals: %r" % sys._getframe(1).f_locals)
		#a = B.a  # wont find `B` here
		_B = sys._getframe(1).f_locals["B"]
		a = _B.a

print("A.C.a = %r" % A.C.a)

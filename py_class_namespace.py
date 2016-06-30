class A:
	class B:
		a = 1
	class C:
		a = B.a  # wont find `B` here


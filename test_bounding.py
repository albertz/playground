class Foo(object):
	def f(self, x): print "f:", self, x

obj = Foo()
obj.f(1)

print type(obj.f), dir(obj.f), obj.f.im_class, obj.f.im_self, obj.f.im_func

obj.f.im_func(42, 2)

print obj.f.__class__

#import types
#print types.instancemethod

class Foo:
    def __call__(self, x):
        print("Original call")
        return x * 2


foo = Foo()
foo(3)

foo.__call__ = lambda x: print("Overwritten foo call") or x * 3
foo(3)  # still original call!

Foo.__call__ = lambda self, x: print("Overwritten Foo call") or x * 4
foo(3)  # now overwritten call

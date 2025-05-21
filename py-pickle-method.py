
import pickle


class Foo:
    def __init__(self, v):
        self.value = v

    def func(self):
        return self.value
    
x = Foo(42)

f = pickle.loads(pickle.dumps(x.func))
assert f() == 42

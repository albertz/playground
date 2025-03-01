"""
Test exception during pickling and/or unpickling

https://github.com/python/cpython/issues/130621
https://discuss.python.org/t/pickle-exception-handling-could-state-object-path/82395
"""

import argparse
import pickle
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--exc-in-getstate", action="store_true")
arg_parser.add_argument("--exc-in-setstate", action="store_true")
args = arg_parser.parse_args()


print("Python:", sys.version)


class MyClass:
    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        if args.exc_in_getstate:
            raise Exception("getstate")
        return self.value

    def __setstate__(self, state):
        if args.exc_in_setstate:
            raise Exception("setstate")
        self.value = state


obj = MyClass(42)
d = {"a": {"b": {"c": obj}, "d": obj}}

s = pickle.dumps(d)
d2 = pickle.loads(s)

obj2 = d2["a"]["b"]["c"]
assert obj2.value == obj2.value
print("OK")

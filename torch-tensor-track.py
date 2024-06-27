"""
Track tensor going out-of-scope
"""

import gc
import torch
import weakref


def main():
    func()
    gc.collect()
    print("After func()")


class _TensorHandle:
    def __init__(self, tensor):
        self.tensor_ref = weakref.ref(tensor)  # not really used here...

    def __del__(self):
        print("out-of-scope")


def func():
    x = torch.zeros(2, 3)
    x._my_handle = _TensorHandle(x)
    return x


if __name__ == "__main__":
    main()

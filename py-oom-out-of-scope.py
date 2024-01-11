
"""
https://github.com/python/cpython/issues/113939
https://github.com/python/cpython/pull/113940
"""

import traceback
import gc


class Obj:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Obj({self.name!r})"

    def __del__(self):
        print("del", self)


def deep(i: int):
    a = Obj(f"a, i={i}")
    if i == 2:
        raise Exception(f"exception at i={i}")
    print(a)


def func():
    for i in range(5):
        gc.collect()
        print("** i:", i)
        try:
            deep(i)
        except Exception as exc:
            print("caught", exc)
            print_tb(exc.__traceback__)
            # traceback.clear_frames(prev_exc.__traceback__)
            clear_tb(exc.__traceback__)
            continue  # continue with next i
        print("deep", i, "done")


def print_tb(tb):
    print("Call stack:")
    while tb:
        frame_i = tb.tb_frame.f_locals.get("i")
        print(f"  {tb.tb_frame.f_code.co_name}: i={frame_i}")
        tb = tb.tb_next


def clear_tb(tb):
    print("Clearing stack:")
    while tb:
        print(tb.tb_frame)
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            print("  cannot clear?")
        else:
            print("  cleared")
            # Using this code triggers that the ref actually goes out of scope, otherwise it does not!
            # print("  now:", tb.tb_frame.f_locals)
        tb = tb.tb_next


if __name__ == '__main__':
    func()
    print("exit")

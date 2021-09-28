

# https://github.com/rwth-i6/returnn_common/issues/16#issuecomment-929030603

from contextlib import contextmanager
import better_exchook


@contextmanager
def loop1(vs):
  print("enter", vs)
  yield
  print("exit", vs)


def test1():
  a, b = 1, 2
  with loop1(locals()):
    b = b + 1


@contextmanager
def loop2(vs):
  print("enter", vs())
  yield
  print("exit", vs())


def test2():
  frame = better_exchook.get_current_frame()
  a, b = 1, 2
  with loop2(lambda: frame.f_locals):
    b = b + 1
  locals()  # mark b as used


@contextmanager
def loop3(vs):
  print("enter", vs)

  class _Loop:
    def __init__(self):
      self.new_vs = None

    def exit(self, vs_):
      assert self.new_vs is None and vs_ is not None
      self.new_vs = vs_

  _loop = _Loop()
  yield _loop
  assert _loop.new_vs is not None
  print("exit", _loop.new_vs)


def test3():
  a, b = 1, 2
  with loop3(locals()) as loop:
    b = b + 1
    loop.exit(locals())


def main():
  test3()


if __name__ == '__main__':
  main()

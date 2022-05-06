"""
https://youtrack.jetbrains.com/issue/PY-54128
"""


def _foo(x):
  assert False


def _main():
  tuple(_foo(a) for a in [1, 2, 3])


if __name__ == '__main__':
  _main()

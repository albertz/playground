

from typing import List


def foo(ls: List[int]):
  # https://youtrack.jetbrains.com/issue/PY-42681
  # https://youtrack.jetbrains.com/issue/PY-30175
  ls_ = list(enumerate(ls))  # PyCharm 2021: ls_: List[int]. wrong!
  print(ls_)

  ls_ = [a for a in enumerate(ls)]  # PyCharm 2021: ls_: List[Tuple[int, int]]. correct
  print(ls_)

  # https://youtrack.jetbrains.com/issue/PY-50448
  ls_ = [(i, a) for (i, a) in enumerate(ls)]  # PyCharm 2021: ls_: List[Tuple[Any, Any]]. incomplete
  print(ls_)


foo([1, 2, 3])

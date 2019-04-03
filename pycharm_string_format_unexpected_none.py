

import typing


class A:
  def __init__(self):
    self.num = None  # type: typing.Optional[int]

  def set_num(self, n):
    """
    :param int n:
    """
    self.num = n

  def hello(self):
    if self.num is not None:
      # https://youtrack.jetbrains.com/issue/PY-35125
      print("Hello %i" % self.num)  # warning: Unexpected type None



import sys
import io


def f_typing(s):
  """
  :param typing.TextIO s:
  """


def f_io(s):
  """
  :param io.TextIOBase s:
  """


string_stream = io.StringIO()
file_stream = open("/dev/null")

f_typing(sys.stdout)
f_typing(string_stream)
f_typing(file_stream)
f_typing("wrong")  # (correct) warning


f_io(sys.stdout)  # warning: Expected type 'TextIOBase', got 'TextIO' instead
f_io(string_stream)
f_io(file_stream)  # warning: Expected type 'TextIOBase', got 'TextIO' instead
f_io("wrong")  # (correct) warning

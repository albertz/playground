

import numpy


state = numpy.random.RandomState(1)
# https://youtrack.jetbrains.com/issue/PY-35164
print(state.rand())  # warning: parameter 'd0'/'d1' unfilled

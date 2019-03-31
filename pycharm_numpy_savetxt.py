
import numpy

# https://youtrack.jetbrains.com/issue/PY-35025
x = numpy.array([42])
numpy.savetxt("file.txt", x)  # warning: expected type 'int', got 'ndarray' instead

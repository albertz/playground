"""
Test

https://youtrack.jetbrains.com/issue/PY-60159/Incorrect-type-inference-from-assert-which-includes-...-is-None
"""

a = 3
b = 4

assert (a is None) == (b is None)

# Incorrect warning: Cannot find reference 'bit_count' in 'None'
print(a.bit_count())

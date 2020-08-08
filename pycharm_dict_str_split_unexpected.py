
"""
https://youtrack.jetbrains.com/issue/PY-43916
"""

s = "a=b,c=d"
opts = dict([opt.split("=", 1) for opt in s.split(",")])
print(opts)



# https://youtrack.jetbrains.com/issue/PY-47568

x = "abc"

x = len(x)

assert isinstance(x, int)



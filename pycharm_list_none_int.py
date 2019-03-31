

# https://youtrack.jetbrains.com/issue/PY-35023
# Warning:
# Expected type 'List[None]' (matched generic type 'List[_T]'),
# got 'List[int]' instead.
ls = [None] + [1, 2, 3]

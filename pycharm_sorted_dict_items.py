
"""
https://youtrack.jetbrains.com/issue/PY-43912
"""

import typing


d = {"abc": [1, 2, 3]}
d_ = sorted(d.items())  # type: typing.List[typing.Tuple[str,typing.List[int]]]
print(d_)

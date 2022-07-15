
# https://youtrack.jetbrains.com/issue/PY-36317/False-positive-Parameterized-generics-cannot-be-used-with-class-or-instance-checks

cls_dict = {"str": str}
cls = cls_dict["str"]
s = cls("abc")

# PyCharm marks cls as error, prints:
# "Parameterized generics cannot be used with instance and class checks"
assert isinstance(s, cls)

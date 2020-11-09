

"""
https://youtrack.jetbrains.com/issue/PY-45376
"""


import importlib

sys = importlib.import_module("sys")

print(sys.version)  # warning: Unresolved attribute reference 'version' for class 'ModuleType'


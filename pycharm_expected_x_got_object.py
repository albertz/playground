

def foo():
    """
    :rtype: object
    """
    return {}


def bar(x):
    """
    :param dict x:
    """


y = foo()
# warning: Expected type 'dict', got 'object' instead
# https://youtrack.jetbrains.com/issue/PY-35270
bar(y)


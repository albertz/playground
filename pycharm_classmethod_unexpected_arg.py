
import threading
PY3 = True


def f():
    """
    https://youtrack.jetbrains.com/issue/PY-34984
    """
    if PY3:
        # noinspection PyPep8Naming
        Condition = threading.Condition
    else:
        # noinspection PyPep8Naming,PyUnresolvedReferences
        Condition = threading.SomethingSomethingDoesNotExist
    cond_wait = Condition.wait

    cond = Condition()
    # https://youtrack.jetbrains.com/issue/PY-34984
    cond_wait(cond, timeout=0.1)  # warning: Unexpected argument


def g():
    """
    https://youtrack.jetbrains.com/issue/PY-34983
    """
    # warning: Variable in function should be lowercase
    # https://youtrack.jetbrains.com/issue/PY-34983
    # https://youtrack.jetbrains.com/issue/PY-28833
    Condition = threading.Condition

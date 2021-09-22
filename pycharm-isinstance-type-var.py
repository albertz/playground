

from typing import TypeVar, Type, Iterable

T = TypeVar('T')
T_type = Type[T]


# https://youtrack.jetbrains.com/issue/PY-31788
def check_instance(obj, type_: Type[T]) -> T:
    assert isinstance(obj, type_)
    return None


a_str = check_instance("foo", str)


# https://youtrack.jetbrains.com/issue/PY-50828
def check_instance_(obj, type_):
    """
    :param obj:
    :param type[T] type_:
    :rtype: T
    """
    assert isinstance(obj, type_)
    return obj


a_str_ = check_instance_("foo", str)


# https://youtrack.jetbrains.com/issue/PY-32860
def check_instance_tuple(obj, types: Iterable[Type[T]]) -> T:
    types = tuple(types)
    assert isinstance(obj, types)
    return obj


a_str__ = check_instance_tuple("foo", [str])


class A:
    pass


class B:
    pass


if __name__ == '__main__':
    a = A()
    print(check_instance(a, A))
    print(check_instance(a, B))
    print(check_instance_tuple(a, (A, B)))
    print(check_instance_tuple(a, (B,)))

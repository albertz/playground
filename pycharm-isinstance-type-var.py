

from typing import TypeVar, Type, Iterable

T = TypeVar('T')


# https://youtrack.jetbrains.com/issue/PY-31788
def check_instance(obj, type_: Type[T]) -> T:
    if isinstance(obj, type_):
        return obj
    return None


# https://youtrack.jetbrains.com/issue/PY-32860
def check_instance_tuple(obj, types: Iterable[Type[T]]) -> T:
    types = tuple(types)
    if isinstance(obj, types):  # Type variables cannot be used with instance and class checks
        return obj
    return None


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

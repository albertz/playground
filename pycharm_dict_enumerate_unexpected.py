

# https://youtrack.jetbrains.com/issue/PY-35037
# warning:
# unexpected type (enumerate[str]),
# possible types: (Mapping) (Iterable[Tuple[Any, Any]])
dict(enumerate(["a", "b", "c"]))




def create(value):
  """
  :param str value:
  :rtype: int
  """
  return hash(value)


x = {key: create(key) for key in ["a", "b"]}
# inferred type for `x` should be 'Dict[str,int]'
# https://youtrack.jetbrains.com/issue/PY-35038

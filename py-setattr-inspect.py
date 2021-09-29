

class StateHolder:

  def __init__(self):
    self._state = {}

  def __setattr__(self, key, value):
    if key == "_state":
      return super(StateHolder, self).__setattr__(key, value)
    print(f"setattr {key} {value!r}")
    if isinstance(value, str):
      value = int(value)
    self._state[key] = value

  def __getattr__(self, item):
    return self._state.get(item, None)


state = StateHolder()
state.x = 42
print(state.x)
print(state.y)
state.z = "2"
print(state.z)

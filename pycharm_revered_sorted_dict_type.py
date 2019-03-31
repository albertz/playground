
data = {"x": 42}

for x, y in reversed(sorted(data.items())):
  # https://youtrack.jetbrains.com/issue/PY-35026
  print(x, y)

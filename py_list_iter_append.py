

l = [1]

for i in l:
  print("iter:", i)
  if len(l) < 10:
    l.append(i + 1)


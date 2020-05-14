

ls = list(range(10))

for x in ls:
  print(x)
  if x == 2:
    ls.remove(2)
    ls.remove(4)
    ls.insert(0, "xxx")
    ls.insert(7, "yyy")


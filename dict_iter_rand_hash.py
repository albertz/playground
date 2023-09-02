
import random


rnd = random.Random(42)


class FunnyObj:
    def __hash__(self):
        return rnd.randint(0, 1)

    def __eq__(self, other):
        return rnd.randint(0, 1) == 0


d = {}

for i in range(1000):
    d[FunnyObj()] = i

print(list(d.items()))
print(d[FunnyObj()])

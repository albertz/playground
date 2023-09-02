
class FunnyObj:
    def __init__(self, inner):
        self.inner = inner

    def __repr__(self):
        return f"FunnyObj({self.inner})"

    def __hash__(self):
        return hash(self.inner)

    def __eq__(self, other):
        return self.inner == other.inner

    def __lt__(self, other):
        return self.inner < other.inner


a = FunnyObj(1)
b = FunnyObj(2)
d = {a: 1, b: 2}

print(list(d.items()))
print(d[FunnyObj(2)])

b.inner = 3
print(list(d.items()))
print(d.get(FunnyObj(2)))
print(d.get(FunnyObj(3)))

for k in d:
    try:
        print(k, ":", d[k])
    except KeyError:  # should not happen normally, but due to changed hash/identity, it might happen here
        print(k, ":", "KeyError")

# might run into same problem, and then raise:
#   RuntimeError: Dictionary was modified during iteration over it
import tree
print(tree.flatten(d))

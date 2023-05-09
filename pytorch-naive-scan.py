"""
PyTorch version

https://github.com/google/jax/issues/15906
"""


import torch
import psutil
import matplotlib.pyplot as plt
import os
import time


batch_dim = 10
feature_dim = 5


def test_scan(time_dim: int):
    xs = torch.rand((time_dim, batch_dim, feature_dim), requires_grad=True)
    ys = torch.zeros((time_dim, batch_dim, feature_dim))

    for i in range(time_dim):
        x = xs[i]
        x_ = x * (i ** 0.5)
        ys[i] = x_

    y = ys.sum()

    mem = psutil.Process().memory_info().rss
    print(f"n={n}, mem={mem}")
    memory_per_n.append(mem)

    y.backward()
    return xs.grad


ns = [5, 10, 20, 50, 100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7_500, 10_000, 15_000, 20_000, 35_000, 50_000]
memory_per_n = []
runtime_per_n = []


for n in ns:
    start = time.time()
    test_scan(n)
    runtime = time.time() - start
    print(f"n={n}, runtime={runtime}")
    runtime_per_n.append(runtime)


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('n')
ax1.set_ylabel('runtime', color=color)
ax1.plot(ns, runtime_per_n, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('memory', color=color)
ax2.plot(ns, memory_per_n, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

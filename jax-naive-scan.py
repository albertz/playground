"""
Testing JAX naive scan implementation

https://github.com/google/jax/issues/15906
"""


import jax
import psutil
import matplotlib.pyplot as plt
import os
import time
from jax.experimental import host_callback as hcb


batch_dim = 10
feature_dim = 5


def test_scan(time_dim: int):
    def func(xs):
        def body(state, _):
            i, ys = state
            x = xs[i]
            x_ = x * (i ** 0.5)
            ys = jax.lax.dynamic_update_index_in_dim(ys, x_, i, axis=0)
            return (i + 1, ys), None

        i = 0
        ys = jax.numpy.zeros((time_dim, batch_dim, feature_dim))
        (_, ys), _ = jax.lax.scan(body, init=(i, ys), xs=None, length=time_dim)

        y = jax.numpy.sum(ys)
        y = id_with_check(y)
        return y

    rnd_key = jax.random.PRNGKey(42)
    xs = jax.random.uniform(rnd_key, (time_dim, batch_dim, feature_dim))
    grad_xs = jax.grad(func)(xs)
    return grad_xs


@jax.custom_vjp
def id_with_check(x):
  return x

def _id_with_check_fwd(x):
  return id_with_check(x), None

def _id_with_check_bwd(res, g):
    hcb.id_tap(_measure_mem, ())
    hcb.barrier_wait()
    return (g,)


def _measure_mem(*_):
    mem = psutil.Process().memory_info().rss
    print(f"n={n}, mem={mem}")
    memory_per_n.append(mem)


id_with_check.defvjp(_id_with_check_fwd, _id_with_check_bwd)


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update('jax_platform_name', 'cpu')


ns = [5, 10, 20, 50, 100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7_500, 10_000, 15_000, 20_000, 35_000, 50_000]
memory_per_n = []
runtime_per_n = []


for n in ns:
    start = time.time()
    y = test_scan(n)
    y.block_until_ready()
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

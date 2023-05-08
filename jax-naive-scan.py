"""
Testing JAX naive scan implementation

https://github.com/google/jax/issues/15906
"""


import jax
import psutil
import matplotlib.pyplot as plt
import os
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
    ys.append(mem)


id_with_check.defvjp(_id_with_check_fwd, _id_with_check_bwd)


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update('jax_platform_name', 'cpu')


xs = list(range(5, 10_000, 500))
ys = []


for n in xs:
    y = test_scan(n)
    y.block_until_ready()


fig, ax = plt.subplots()
ax.plot(xs, ys)
plt.show()

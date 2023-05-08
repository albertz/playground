"""
Testing JAX naive scan implementation

https://github.com/google/jax/issues/15906
"""


import jax
import psutil
import matplotlib.pyplot as plt
import os


batch_dim = 10
feature_dim = 5


def test_scan(time_dim: int):
    def func(xs):
        def body(state):
            i, ys = state
            x = xs[i]
            x_ = x * (i ** 0.5)
            ys = jax.lax.dynamic_update_index_in_dim(ys, x_, i, axis=0)
            return i + 1, ys

        def cond(state):
            i, _ = state
            return i < time_dim

        i = 0
        ys = jax.numpy.zeros((time_dim, batch_dim, feature_dim))
        _, ys = jax.lax.while_loop(cond, body, (i, ys))

        y = jax.numpy.sum(ys)
        return y

    rnd_key = jax.random.PRNGKey(42)
    xs = jax.random.uniform(rnd_key, (time_dim, batch_dim, feature_dim))
    grad_xs = jax.grad(func)(xs)
    print(grad_xs)
    return grad_xs


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update('jax_platform_name', 'cpu')


xs = list(range(5, 100, 10))
ys = []


for n in xs:
    y = test_scan(n)
    y.block_until_ready()
    mem = psutil.Process().memory_info().rss
    print(f"n={n}, mem={mem}")
    ys.append(mem)


fig, ax = plt.subplots()
ax.plot(xs, ys)
plt.show()

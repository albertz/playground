import jax.numpy as jnp

n = 3
k = 4
m = 5
a = jnp.ones((n, k))
b = jnp.ones((k, m))

# (n,k),(k,m)->(n,m)
y = jnp.matmul(a, b)
print(y)

import jax.numpy as np
import jax
import numpy as onp
import matplotlib.pyplot as plt
m = 4
b = 2

x = np.linspace(0,1,100)

### Generate data ###

def create_data(n, std):
    # n = number of points
    # std = standard deviation

    key = jax.random.key(0)

    def eval_f(x):
        # y = np.sin(2 * np.pi * x)
        y = x * m + b
        return y

    x = onp.linspace(0, 1, n)
    y = jax.vmap(eval_f)(x)
    y += std * jax.random.normal(key, shape=y.shape)

    return x, y

n, std = 100, .2

x, y = create_data(n, std)

fig, ax = plt.subplots()
ax.plot(x, y, "ro")
plt.show()




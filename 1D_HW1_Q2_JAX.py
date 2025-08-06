import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt


def quadratic_1D_lagrange(xi):
    N1 = 1 / 2 * (xi) * (xi - 1)
    N2 = - (xi - 1) * (xi + 1)
    N3 = 1 / 2 * (xi + 1) * (xi)
    return np.array([N1, N2, N3])


def k(x):
    return 0.01 * x ** 2 + 0.5

shape_fn_grads = jax.jacobian(quadratic_1D_lagrange)

xi = 0.25

NoE = 3
NpE = 3
NoN = NoE * (NpE-1) + 1
L = 0.6


nodal_indices = np.arange(0,NoN,1)

x_array = np.linspace(0,L,NoN)

x_batched = []


for i in range(NoE):
    element_nodal_indices = nodal_indices[i*(NpE-1):i*(NpE-1)+NpE]
    elements_coords = x_array[i*(NpE-1):i*(NpE-1)+NpE]
    x_batched.append(np.array(elements_coords))
    print(elements_coords)

x_batched = np.array(x_batched)
print(shape_fn_grads(xi))

x = np.array([0,0.1,0.2]).reshape(-1,1)
xi = 0.25

def integrand(xi, x, lagrange):
    grads = shape_fn_grads(xi).reshape(-1,1)
    vals = lagrange(xi).reshape(-1,1)
    J = x.T @ grads
    grads_physical = grads @ np.linalg.inv(J)
    k_xi = k(x.T @ vals)
    return k_xi * (grads_physical @ grads_physical.T) * np.linalg.det(J)


integrand(xi,x, quadratic_1D_lagrange)
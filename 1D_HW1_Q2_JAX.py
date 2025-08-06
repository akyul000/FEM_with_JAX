import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import gauss_quadrature as gq


def linear_1D_lagrange(xi):
    N1 = - 1 / 2  * (xi - 1)
    N2 = 1 / 2 * (xi + 1)
    return np.array([N1, N2])

def quadratic_1D_lagrange(xi):
    N1 = 1 / 2 * (xi) * (xi - 1)
    N2 = - (xi - 1) * (xi + 1)
    N3 = 1 / 2 * (xi + 1) * (xi)
    return np.array([N1, N2, N3])


def k(x):
    return 0.01 * x ** 2 + 0.5


def T_exact ( x ) :
    T = 50 * np.sqrt (2) * ( -np.arctan ( np.sqrt(2) ) + np.arctan ( np.sqrt(2) * x / 10) )
    return T
###############
# Input Params
NoE = 3
NpE = 2
NoN = NoE * (NpE-1) + 1
L = 10
F_N = -5
lagrange = linear_1D_lagrange
###############
shape_fn_grads = jax.jacobian(lagrange)
nodal_indices = np.arange(0,NoN,1)
x_array = np.linspace(0,L,NoN)
x_batched = []
for i in range(NoE):
    element_nodal_indices = nodal_indices[i*(NpE-1):i*(NpE-1)+NpE]
    elements_coords = x_array[i*(NpE-1):i*(NpE-1)+NpE]
    x_batched.append(np.array(elements_coords))
x_batched = np.array(x_batched)
x_analytical = np.linspace(0,L,1000)




def integrand(xi, x, lagrange):
    grads = shape_fn_grads(xi).reshape(-1,1)
    vals = lagrange(xi).reshape(-1,1)
    J = x.T @ grads
    grads_physical = grads @ np.linalg.inv(J)
    k_xi = k(x.T @ vals)
    return k_xi * (grads_physical @ grads_physical.T) * np.linalg.det(J)





def make_integrand_wrapper(X, lagrange):
    """
    Returns a function f(xi, eta) that applies over all elements in the batch.
    X_batch, Y_batch: shape (n_elem, 4, 1)
    """
    def f(xi):
        return jax.vmap(lambda X: integrand(xi, X, lagrange))(X)
    return f  

f = make_integrand_wrapper(x_batched.reshape(NoE,NpE,1), lagrange)
Q = gq.Quadrature(dim=1, p=2, f=f)
ke_all = Q.integrate()  







K = np.zeros((NoN,NoN))
for i in range(NoE):
    element_nodal_inds = nodal_indices[i*(NpE-1):i*(NpE-1)+NpE]
    ke = ke_all[i]  # shape (NpE, NpE)

    idx_j, idx_k = np.meshgrid(element_nodal_inds, element_nodal_inds, indexing='ij')
    K = K.at[idx_j, idx_k].add(ke)


F = np.zeros((NoN, 1))
F = F.at[0].set(F_N)
K_reduced = K[:-1,:-1]
F_reduced = F[:-1]
u_reduced = np.linalg.solve(K_reduced, F_reduced)
u = np.zeros((NoN,1))
u = u.at[:-1].set(u_reduced)


plt.figure()
plt.plot(x_array, u, "ro", label = "Nodal solutions")
plt.plot(x_analytical, T_exact(x_analytical),label="Analytical Solution")
plt.legend()
plt.show()

def evaluate_T_in_element(u_element, x_element, xi_vals, lagrange):
    """
    u_element: shape (NpE, 1) nodal values for the element
    x_element: shape (NpE,) physical coordinates of the element nodes
    xi_vals: parametric points in [-1, 1]
    lagrange: shape function
    """
    N = vmap(lagrange)(xi_vals)  # shape (n_pts, NpE)
    T_vals = N @ u_element.reshape(-1)  # shape (n_pts,)
    
    x_physical = N @ x_element.reshape(-1)
    
    return x_physical, T_vals

xi_dense = np.linspace(-1, 1, 20)  # More points → smoother plot
x_plot = []
T_plot = []

for i in range(NoE):
    inds = nodal_indices[i*(NpE-1):i*(NpE-1)+NpE]
    u_elem = u[inds]  # Nodal values for this element
    x_elem = x_array[inds]  # Physical coordinates

    x_vals, T_vals = evaluate_T_in_element(u_elem, x_elem, xi_dense, lagrange)
    x_plot.append(onp.array(x_vals))
    T_plot.append(onp.array(T_vals))

x_plot = onp.concatenate(x_plot)
T_plot = onp.concatenate(T_plot)

plt.figure(figsize=(8,5))
plt.plot(x_plot, T_plot, label="FEM (interpolated)", lw=2)
plt.plot(x_array, u, "ro", label = "FEM (nodes)")
plt.plot(x_analytical, T_exact(x_analytical), label="Analytical", ls='--')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.legend()
plt.title("Post-processed FEM Solution vs Analytical")
plt.grid(True)
plt.tight_layout()
plt.show()

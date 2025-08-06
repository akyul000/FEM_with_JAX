# import NURBS_helper_functions as NURBS_CARDIAX  
# import numpy as onp
# import jax.numpy as np

# # Input
# p = 2
# P_x = 4
# L = 10
# # x_cps = np.arange()
# # x_array = np.linspace(0,L,NoN)
# cps = np.linspace(0,L,P_x).reshape(-1,1)


# knot_vector = NURBS_CARDIAX.get_open_uniform_knot_vector(P_x, p)



# num_eval = np.linspace(0,1,10)



# for i in range(len(num_eval)):
#     xi = num_eval[i]
#     span_i = NURBS_CARDIAX.find_span(knot_vector, p, xi)
#     N_i, dN_i = NURBS_CARDIAX.bspline_basis_and_derivatives(knot_vector, p, span_i, p)
#     cps_idx = NURBS_CARDIAX.get_control_point_indices(span_i, p)

# print()


# def integrand(xi, x, lagrange):
#     span_i = NURBS_CARDIAX.find_span(knot_vector, p, xi)
#     N_i, dN_i = NURBS_CARDIAX.bspline_basis_and_derivatives(knot_vector, p, span_i, p)
#     cps_idx = NURBS_CARDIAX.get_control_point_indices(span_i, p)

#     grads = dN_i.reshape(-1,1)
#     vals = lagrange(xi).reshape(-1,1)
#     J = x.T @ grads
#     grads_physical = grads @ np.linalg.inv(J)
#     k_xi = k(x.T @ vals)
#     return k_xi * (grads_physical @ grads_physical.T) * np.linalg.det(J)


# import jax.numpy as np
# import numpy as onp
# import matplotlib.pyplot as plt
# from jax import jit, vmap
# import jax

# jax.config.update("jax_enable_x64", True)

# import NURBS_helper_functions as NURBS  # Make sure this includes the required helper functions

# class Quadrature:
#     def __init__(self, dim, p, f):
#         self.dim = dim
#         self.p = p
#         self.f = f  # f should take mapped xi values

#     def rule(self):
#         if self.p <= 1:
#             self.n = 1
#             self.x = np.array([0.])
#             self.w = np.array([2.])
#         elif self.p <= 3:
#             self.n = 2
#             self.x = np.array([1/np.sqrt(3), -1/np.sqrt(3)])
#             self.w = np.array([1., 1.])
#         elif self.p <= 5:
#             self.n = 3
#             self.x = np.array([np.sqrt(0.6), 0., -np.sqrt(0.6)])
#             self.w = np.array([5/9, 8/9, 5/9])
#         else:
#             raise NotImplementedError("Higher-order quadrature not implemented.")

#     def integrate(self, a=0., b=1.):
#         self.rule()
#         gauss_quadrature = 0.

#         if self.dim == 1:
#             for i in range(self.n):
#                 xi_hat = self.x[i]
#                 xi_phys = ((b - a) / 2) * xi_hat + (a + b) / 2
#                 weight = self.w[i] * (b - a) / 2
#                 gauss_quadrature += weight * self.f(xi_phys)
#         else:
#             raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")

#         return gauss_quadrature


# # Problem-specific functions
# def k(x):
#     return 0.01 * x**2 + 0.5

# def T_exact(x):
#     return 50 * np.sqrt(2) * (-np.arctan(np.sqrt(2)) + np.arctan(np.sqrt(2) * x / 10))


# # Parameters
# p = 2                   # B-spline degree
# n_cps = 4               # Number of control points
# L = 10.0                # Length of domain

# # Control points (1D uniform)
# cps = np.linspace(0, L, n_cps).reshape(-1, 1)

# # Knot vector
# knot_vector = NURBS.get_open_uniform_knot_vector(n_cps, p)
# unique_knots = np.unique(knot_vector)
# num_elem = len(unique_knots) - 1

# n_basis = n_cps
# K = np.zeros((n_basis, n_basis))
# F = np.zeros((n_basis, 1))


# # Loop over elements (knot spans)
# for e in range(num_elem):
#     a = unique_knots[e]
#     b = unique_knots[e + 1]
#     if b - a < 1e-10:
#         continue  # skip degenerate spans

#     def integrand(xi):
#         span = NURBS.find_span(knot_vector, p, xi)
#         N_i, dN_i = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, p)
#         cps_idx = NURBS.get_control_point_indices(span, p)
        
#         N_i = np.array(N_i).reshape(-1, 1)
#         dN_i = np.array(dN_i).reshape(-1, 1)

#         x_elem = cps[cps_idx]
#         J = x_elem.T @ dN_i
#         dx_dxi = J[0, 0]
#         dN_dx = dN_i / dx_dxi
#         x_val = (x_elem.T @ N_i)[0, 0]
#         k_val = k(x_val)

#         ke = k_val * (dN_dx @ dN_dx.T) * dx_dxi
#         return ke, cps_idx

#     # Inner function for Quadrature
#     def integrand_matrix(xi):
#         ke, _ = integrand(xi)
#         return ke

#     quad = Quadrature(dim=1, p=p, f=integrand_matrix)
#     ke = quad.integrate(a, b)

#     _, cps_idx = integrand((a + b) / 2)  # Any xi works to get local indices
#     idx_j, idx_k = np.meshgrid(cps_idx, cps_idx, indexing='ij')
#     K = K.at[idx_j, idx_k].add(ke)

# # Neumann BC at x = 0 → q = -5 W/m^2
# F = F.at[0].set(-5.0)

# # Dirichlet BC at x = L → T = 0
# # Assume last control point is associated with x = L
# K_reduced = K[:-1, :-1]
# F_reduced = F[:-1]

# u_reduced = np.linalg.solve(K_reduced, F_reduced)
# u = np.zeros((n_basis, 1))
# u = u.at[:-1].set(u_reduced)


# x_vals = []
# u_vals = []

# eval_xi = np.linspace(0, 1, 200)
# for xi in eval_xi:
#     span = NURBS.find_span(knot_vector, p, xi)
#     N_i, _ = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, p)
#     cps_idx = NURBS.get_control_point_indices(span, p)
#     N_i = np.array(N_i).reshape(1, -1)

#     u_local = u[cps_idx].reshape(-1)
#     x_local = cps[cps_idx].reshape(-1)

#     T_val = N_i @ u_local
#     x_val = N_i @ x_local

#     u_vals.append(T_val[0])
#     x_vals.append(x_val[0])

# x_vals = onp.array(x_vals)
# u_vals = onp.array(u_vals)

# plt.figure()
# plt.plot(x_vals, u_vals, label='IGA Solution', lw=2)
# plt.plot(x_vals, T_exact(x_vals), '--', label='Analytical Solution')
# plt.scatter(cps, u, color='r', label='Control Points')
# plt.xlabel("x")
# plt.ylabel("Temperature")
# plt.title("IGA vs Analytical - 1D Heat Conduction")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from jax import jit
import jax

jax.config.update("jax_enable_x64", True)

import NURBS_helper_functions as NURBS  # Ensure this includes all necessary B-spline functions

class Quadrature:
    def __init__(self, dim, p, f):
        self.dim = dim
        self.p = p
        self.f = f  # f should take mapped xi values

    def rule(self):
        if self.p <= 1:
            self.n = 1
            self.x = np.array([0.])
            self.w = np.array([2.])
        elif self.p <= 3:
            self.n = 2
            val = 1.0 / np.sqrt(3.0)
            self.x = np.array([-val, val])
            self.w = np.array([1.0, 1.0])
        elif self.p <= 5:
            val = np.sqrt(0.6)
            self.n = 3
            self.x = np.array([-val, 0.0, val])
            self.w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        else:
            raise NotImplementedError("Higher-order quadrature not implemented.")

    def integrate(self, a=0., b=1.):
        self.rule()
        result = 0.

        if self.dim == 1:
            for i in range(self.n):
                xi_hat = self.x[i]
                xi_phys = ((b - a) / 2) * xi_hat + (a + b) / 2
                weight = self.w[i] * (b - a) / 2
                result += weight * self.f(xi_phys)
        else:
            raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")

        return result

# Problem-specific functions
def k(x):
    return 0.01 * x**2 + 0.5

def T_exact(x):
    return 50 * np.sqrt(2) * (-np.arctan(np.sqrt(2)) + np.arctan(np.sqrt(2) * x / 10))

# Parameters
p = 2
n_cps = 8
L = 10.0

cps = np.linspace(0, L, n_cps).reshape(-1, 1)
knot_vector = NURBS.get_open_uniform_knot_vector(n_cps, p)
unique_knots = np.unique(knot_vector)
num_elem = len(unique_knots) - 1

n_basis = n_cps
K = np.zeros((n_basis, n_basis))
F = np.zeros((n_basis, 1))

# Loop over knot spans
tol = 1e-10
for e in range(num_elem):
    a = unique_knots[e]
    b = unique_knots[e + 1]
    if b - a < tol:
        continue

    def integrand(xi):
        span = NURBS.find_span(knot_vector, p, xi)
        N_i, dN_i = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
        cps_idx = NURBS.get_control_point_indices(span, p)

        N_i = np.array(N_i).reshape(-1, 1)
        dN_i = np.array(dN_i).reshape(-1, 1)

        x_elem = cps[cps_idx]
        J = x_elem.T @ dN_i
        dx_dxi = J[0, 0]
        dN_dx = dN_i / dx_dxi
        x_val = (x_elem.T @ N_i)[0, 0]
        k_val = k(x_val)

        ke = k_val * (dN_dx @ dN_dx.T) * dx_dxi
        return ke, cps_idx

    def integrand_matrix(xi):
        ke, _ = integrand(xi)
        return ke

    quad = Quadrature(dim=1, p=p, f=integrand_matrix)
    ke = quad.integrate(a, b)

    _, cps_idx = integrand((a + b) / 2)
    idx_j, idx_k = np.meshgrid(cps_idx, cps_idx, indexing='ij')
    K = K.at[idx_j, idx_k].add(ke)

# Neumann BC at x = 0 (flux q = -5)
F = F.at[0].set(-5.0)

# Dirichlet BC at x = L (T = 0)
K_reduced = K[:-1, :-1]
F_reduced = F[:-1]

u_reduced = np.linalg.solve(K_reduced, F_reduced)
u = np.zeros((n_basis, 1))
u = u.at[:-1].set(u_reduced)

# Post-process for visualization
x_vals = []
u_vals = []

xi_vals = np.linspace(0, 1, 200)
for xi in xi_vals:
    span = NURBS.find_span(knot_vector, p, xi)
    N_i, _ = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
    cps_idx = NURBS.get_control_point_indices(span, p)
    N_i = np.array(N_i).reshape(1, -1)

    u_local = u[cps_idx].reshape(-1)
    x_local = cps[cps_idx].reshape(-1)

    u_val = N_i @ u_local
    x_val = N_i @ x_local

    u_vals.append(u_val[0])
    x_vals.append(x_val[0])

x_vals = onp.array(x_vals)
u_vals = onp.array(u_vals)

plt.figure()
plt.plot(x_vals, u_vals, label='IGA Solution', lw=2)
plt.plot(x_vals, T_exact(x_vals), '--', label='Analytical Solution')
plt.scatter(cps, u, color='r', label='Control Points')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("IGA vs Analytical - 1D Heat Conduction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
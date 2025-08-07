
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from jax import jit
import jax

jax.config.update("jax_enable_x64", True)
import general_gauss_quadrature as ggq
import NURBS_helper_functions as NURBS 

# Problem-specific functions
def k(x):
    return 0.01 * x**2 + 0.5

def T_exact(x):
    return 50 * np.sqrt(2) * (-np.arctan(np.sqrt(2)) + np.arctan(np.sqrt(2) * x / 10))

# Parameters
p = 2
n_cps = 6
L = 10.0
F_N = -5
num_xi_array_for_post_processing = 50


cps = np.linspace(0, L, n_cps).reshape(-1, 1)
knot_vector = NURBS.get_open_uniform_knot_vector(n_cps, p)
unique_knots = np.unique(knot_vector)
num_elem = len(unique_knots) - 1

n_basis = n_cps
K = np.zeros((n_basis, n_basis))
F = np.zeros((n_basis, 1))

def integrand(xi, knot_vector, p, cps):
    span = NURBS.find_span(knot_vector, p, xi)
    N_i, dN_i = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
    cps_idx = NURBS.get_control_point_indices(span, p)

    N_i = np.array(N_i).reshape(-1, 1)
    dN_i = np.array(dN_i).reshape(-1, 1)

    x_elem = cps[cps_idx]
    J = x_elem.T @ dN_i
    grads_physical = dN_i @ np.linalg.inv(J)

    x_val = (x_elem.T @ N_i)[0, 0]
    k_val = k(x_val)

    ke = k_val * (grads_physical @ grads_physical.T) * np.linalg.det(J)
    return ke




def integrand_wrapper(knot_vector, p, cps):
    def f(xi):
        ke = integrand(xi, knot_vector, p, cps)
        return ke
    return f
f_v = integrand_wrapper(knot_vector, p, cps)

# Calculate element stiffness matrices and assemble the global system
# Loop over knot spans
cps_batched = []
for i in range(num_elem):
    left_xi, right_xi = unique_knots[i], unique_knots[i+1]
    xi_avg_for_indexing = (left_xi + right_xi) / 2
    span_i = NURBS.find_span(knot_vector, p, xi_avg_for_indexing)
    cps_idx = NURBS.get_control_point_indices(span_i, p=p)
    quadrature = ggq.Quadrature(dim=1, p=p, f=f_v)
    ke = quadrature.integrate(left_xi, right_xi)

    idx_j, idx_k = np.meshgrid(cps_idx, cps_idx, indexing='ij')
    K = K.at[idx_j, idx_k].add(ke)

#################### Apply BCs and Solve ####################

# Neumann BC at x = 0 (flux q = -5)
F = F.at[0].set(F_N)
# Dirichlet BC at x = L (T = 0)
K_reduced = K[:-1, :-1]
F_reduced = F[:-1]
u_reduced = np.linalg.solve(K_reduced, F_reduced)
u = np.zeros((n_basis, 1))  
u = u.at[:-1].set(u_reduced)


#################### Post-Process ####################
# Post-process for visualization
x_vals = []
u_vals = []


xi_post_pro = np.linspace(0,1, num_xi_array_for_post_processing)

for xi in xi_post_pro:
    span = NURBS.find_span(knot_vector, p, xi)
    N_i, _ = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
    cps_idx = NURBS.get_control_point_indices(span, p)
    N_i = np.array(N_i).reshape(-1,1) # onp  to np array

    u_local = u[cps_idx].reshape(-1)
    x_local = cps[cps_idx].reshape(-1)

    u_val = u_local @ N_i
    x_val = x_local @ N_i

    u_vals.append(u_val[0])
    x_vals.append(x_val[0])


x_post_processed = np.array(x_vals).reshape(-1)
u_post_processed = np.array(u_vals).reshape(-1)


print(np.linalg.norm(u_post_processed- T_exact(x_post_processed)))

plt.figure()
plt.plot(x_post_processed, u_post_processed, label='IGA Solution', lw=2)
plt.plot(x_post_processed, T_exact(x_post_processed), '--', label='Analytical Solution')
plt.scatter(cps, u, color='r', label='Control Points')
plt.xlabel("x")
plt.ylabel("Temperature")
plt.title("IGA vs Analytical - 1D Heat Conduction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


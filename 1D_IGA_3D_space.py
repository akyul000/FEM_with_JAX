
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
n_cps = 8
L = 10.0
F_N = -5
num_xi_array_for_post_processing = 50

# Define start and end points
P0 = np.array([-14.6252, 3.0416, -33.0945])
P1 = np.array([ -2.5005, 3.5218, -18.5171])

# Discretize n_cps points between P0 and P1
cps = np.linspace(P0, P1, n_cps)  # shape = (n_cps, 3)

# theta = np.linspace(0, np.pi, n_cps)
# cps = np.stack([np.cos(theta), np.sin(theta), theta**2], axis=1)  # control points along a helical path



# cps = np.linspace(0, L, n_cps).reshape(-1, 1)
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

    N_i = np.array(N_i).reshape(-1, 1)      # (n, 1)
    dN_i = np.array(dN_i).reshape(-1, 1)    # (n, 1)

    x_elem = cps[cps_idx]                  # (n, 3)
    tangent = x_elem.T @ dN_i              # (3, 1)
    J_norm = np.linalg.norm(tangent)       # scalar

    grads_physical = dN_i / J_norm         # (n, 1), normalized derivative in physical space

    x_val = (x_elem.T @ N_i).reshape(-1)   # 3D coordinate of current point
    k_val = k(np.linalg.norm(x_val))       # k is now based on arc-length, or replace with k(x_val)

    ke = k_val * (grads_physical @ grads_physical.T) * J_norm
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
    N_i = np.array(N_i).reshape(-1, 1)

    u_local = u[cps_idx].reshape(-1)
    x_local = cps[cps_idx].reshape(-1, 3)

    u_val = u_local @ N_i        # scalar
    x_val = (x_local.T @ N_i).reshape(-1)  # 3D position

    u_vals.append(u_val[0])
    x_vals.append(x_val)

x_post_processed = np.array(x_vals)       # (n, 3)
u_post_processed = np.array(u_vals)       # (n,)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Normalize temperature for colormap
u_norm = (u_post_processed - u_post_processed.min()) / (u_post_processed.max() - u_post_processed.min())

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Convert x_post_processed to separate x, y, z arrays
x, y, z = x_post_processed[:, 0], x_post_processed[:, 1], x_post_processed[:, 2]

# Plot the curve with color mapped to temperature
for i in range(len(x) - 1):
    ax.plot(
        x[i:i+2], y[i:i+2], z[i:i+2],
        color=cm.viridis(u_norm[i]),
        linewidth=3
    )

# Optional: Add scatter points for control points
ax.scatter(cps[:, 0], cps[:, 1], cps[:, 2], c='r', label='Control Points', s=30)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("IGA Temperature Distribution Along 3D Curve")

# Add colorbar
mappable = cm.ScalarMappable(cmap=cm.viridis)
mappable.set_array(u_post_processed)
fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label="Temperature")

plt.tight_layout()
plt.legend()
plt.show()

# Compute arc length for each x value
arc_lengths = np.linalg.norm(np.diff(x_post_processed, axis=0), axis=1)
arc_lengths = np.insert(np.cumsum(arc_lengths), 0, 0.0)

plt.figure()
plt.plot(arc_lengths, u_post_processed, label='IGA Solution', lw=2)
plt.xlabel("Arc Length")
plt.ylabel("Temperature")
plt.title("IGA Temperature Solution Along 3D Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

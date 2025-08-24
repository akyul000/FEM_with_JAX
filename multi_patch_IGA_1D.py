# import jax.numpy as np
# import numpy as onp
# import matplotlib.pyplot as plt
# from jax import jit
# import jax

# jax.config.update("jax_enable_x64", True)

# import general_gauss_quadrature as ggq
# import NURBS_helper_functions as NURBS

# # Problem-specific functions
# def k(x):
#     return 0.01 * x**2 + 0.5

# def T_exact(x):
#     return 50 * np.sqrt(2) * (-np.arctan(np.sqrt(2)) + np.arctan(np.sqrt(2) * x / 10))

# # Parameters
# p = 2
# n_cps_per_patch = 4
# L = 10.0
# interface = 5.0
# F_N = -5
# num_xi_array_for_post_processing = 100

# # Define two patches
# patches = []

# for start, end in [(0.0, interface), (interface, L)]:
#     cps = np.linspace(start, end, n_cps_per_patch).reshape(-1, 1)
#     knot_vector = NURBS.get_open_uniform_knot_vector(n_cps_per_patch, p)
#     patches.append({
#         "cps": cps,
#         "knot_vector": knot_vector,
#         "span_range": (start, end)
#     })

# # Handle shared DOFs: total unique control points
# # Shared point at interface → remove duplicate
# global_cps = onp.vstack([
#     patches[0]["cps"], 
#     patches[1]["cps"][1:]
# ])
# n_basis = global_cps.shape[0]

# K = np.zeros((n_basis, n_basis))
# F = np.zeros((n_basis, 1))

# # Assembly loop per patch
# offset = 0
# for i, patch in enumerate(patches):
#     cps = patch["cps"]
#     knot_vector = patch["knot_vector"]
#     unique_knots = np.unique(knot_vector)
#     num_elem = len(unique_knots) - 1

#     def integrand(xi):
#         span = NURBS.find_span(knot_vector, p, xi)
#         N_i, dN_i = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
#         cps_idx = NURBS.get_control_point_indices(span, p)

#         N_i = np.array(N_i).reshape(-1, 1)
#         dN_i = np.array(dN_i).reshape(-1, 1)

#         x_elem = cps[cps_idx]
#         J = x_elem.T @ dN_i
#         grads_physical = dN_i @ np.linalg.inv(J)

#         x_val = (x_elem.T @ N_i)[0, 0]
#         k_val = k(x_val)
#         ke = k_val * (grads_physical @ grads_physical.T) * np.linalg.det(J)
#         return ke

#     quadrature = ggq.Quadrature(dim=1, p=p, f=integrand)

#     for elem_i in range(num_elem):
#         xi_a, xi_b = unique_knots[elem_i], unique_knots[elem_i + 1]
#         if xi_b == xi_a:
#             continue
#         span_i = NURBS.find_span(knot_vector, p, (xi_a + xi_b) / 2)
#         local_cps_idx = NURBS.get_control_point_indices(span_i, p)

#         ke = quadrature.integrate(xi_a, xi_b)

#         # Map local patch DOFs to global indices
#         global_indices = []
#         for local_idx in local_cps_idx:
#             if i == 1 and local_idx == 0:
#                 global_indices.append(offset - 1)  # Shared point
#             else:
#                 global_indices.append(offset + local_idx)

#         idx_j, idx_k = np.meshgrid(global_indices, global_indices, indexing='ij')
#         K = K.at[idx_j, idx_k].add(ke)

#     offset += len(cps)
#     if i == 1:
#         offset -= 1  # for shared point

# # Apply BCs and solve
# F = F.at[0].set(F_N)  # Neumann at x=0
# K_reduced = K[:-1, :-1]
# F_reduced = F[:-1]
# u_reduced = np.linalg.solve(K_reduced, F_reduced)
# u = np.zeros((n_basis, 1))
# u = u.at[:-1].set(u_reduced)

# # Post-processing
# x_vals, u_vals = [], []
# xi_post = np.linspace(0, 1, num_xi_array_for_post_processing)

# offset = 0
# for i, patch in enumerate(patches):
#     cps = patch["cps"]
#     knot_vector = patch["knot_vector"]

#     for xi in xi_post:
#         span = NURBS.find_span(knot_vector, p, xi)
#         N_i, _ = NURBS.bspline_basis_and_derivatives(knot_vector, p, span, xi)
#         cps_idx = NURBS.get_control_point_indices(span, p)

#         N_i = np.array(N_i).reshape(-1, 1)

#         global_indices = []
#         for local_idx in cps_idx:
#             if i == 1 and local_idx == 0:
#                 global_indices.append(offset - 1)
#             else:
#                 global_indices.append(offset + local_idx)

#         u_local = u[global_indices].reshape(-1)
#         x_local = global_cps[global_indices].reshape(-1)

#         u_val = u_local @ N_i
#         x_val = x_local @ N_i

#         x_vals.append(x_val[0])
#         u_vals.append(u_val[0])

#     offset += len(cps)
#     if i == 1:
#         offset -= 1

# x_post = np.array(x_vals)
# u_post = np.array(u_vals)

# print("Error norm:", np.linalg.norm(u_post - T_exact(x_post)))

# plt.plot(x_post, u_post, label='Multi-Patch IGA', lw=2)
# plt.plot(x_post, T_exact(x_post), '--', label='Analytical')
# plt.scatter(global_cps, u, c='r', label='Control Points')
# plt.xlabel("x")
# plt.ylabel("Temperature")
# plt.grid(True)
# plt.legend()
# plt.title("Multi-Patch IGA (1D Heat)")
# plt.tight_layout()
# plt.show()


"""
Mitral Valve multi-patch + chordae (demo)
Simplified, pedagogical demo showing:
 - One or two 3D IGA patches (degree-1 tensor-product -> hex-like elements)
 - 1D chordae represented as truss elements
 - Coupling using a penalty tie (penalty enforces chord attachment point to solid surface point)
 - Uses JAX for vectorized stiffness assembly / auto-diff friendly structure

Notes:
 - This is a _minimal_ example intended to show structure, not a production-ready solver.
 - Extend by: higher-degree B-splines, Lagrange multipliers for exact constraints, multi-patch interface assembly, nonlinear materials.

Requirements:
 - jax, jaxlib
 - numpy, scipy

Run: `python mitral_valve_multi_patch_chordae_demo.py`

"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------- Utilities: simple linear B-spline basis (degree=1)

def bspline_linear_basis(xi):
    # For degree=1 on knots [0,0,1,1] the basis on [0,1] is: N1 = 1-xi, N2 = xi
    return jnp.array([1 - xi, xi])

# tensor-product evaluation for degree-1 in 3D

def tensor_basis_3d(xi, eta, zeta):
    Nx = bspline_linear_basis(xi)
    Ny = bspline_linear_basis(eta)
    Nz = bspline_linear_basis(zeta)
    # 8 shape functions order: (ix,iy,iz) lexicographic
    N = jnp.zeros(8)
    idx = 0
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                N = N.at[idx].set(Nx[ix] * Ny[iy] * Nz[iz])
                idx += 1
    return N

# derivatives wrt param (for degree=1 simple)

def tensor_basis_3d_derivs(xi, eta, zeta):
    # dN/dxi, dN/deta, dN/dzeta for 8 functions
    Nx = jnp.array([-1.0, 1.0])
    Ny = jnp.array([-1.0, 1.0])
    Nz = jnp.array([-1.0, 1.0])
    dN_dxi = jnp.zeros(8)
    dN_deta = jnp.zeros(8)
    dN_dzeta = jnp.zeros(8)
    idx = 0
    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                dN_dxi = dN_dxi.at[idx].set(Nx[ix] * ( (1+Ny[iy])/2.0 * (1+Nz[iz])/2.0 ))
                # But above is messy; since Nx is constant -1 or 1 and original N = (1-xi or xi)*(1-eta or eta)*(1-zeta or zeta)
                # Simpler: compute analytic derivatives from basis functions directly
                idx += 1
    # Simpler approach: compute derivative numerically small eps
    eps = 1e-6
    def N_at(xi_, eta_, zeta_):
        return tensor_basis_3d(xi_, eta_, zeta_)
    N0 = N_at(xi, eta, zeta)
    Nxi = (N_at(xi+eps, eta, zeta) - N0) / eps
    Neta = (N_at(xi, eta+eps, zeta) - N0) / eps
    Nzeta = (N_at(xi, eta, zeta+eps) - N0) / eps
    return Nxi, Neta, Nzeta

# ----------------------------- Create a simple 3D patch (degree-1) control grid

def make_hexa_patch(nx, ny, nz, origin=(0.0,0.0,0.0), size=(1.0,1.0,1.0)):
    # nodes on a regular grid (control points)
    ox, oy, oz = origin
    sx, sy, sz = size
    xs = jnp.linspace(ox, ox+sx, nx)
    ys = jnp.linspace(oy, oy+sy, ny)
    zs = jnp.linspace(oz, oz+sz, nz)
    pts = jnp.array([ [x,y,z] for z in zs for y in ys for x in xs ])
    # element connectivity for degree-1: elements = (nx-1)*(ny-1)*(nz-1) each with 8 nodes
    elems = []
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                n0 = i + j*nx + k*nx*ny
                # local node numbering consistent with tensor_basis_3d
                conn = [
                    n0 + 0,
                    n0 + 1,
                    n0 + 1 + nx,
                    n0 + 0 + nx,
                    n0 + nx*ny,
                    n0 + 1 + nx*ny,
                    n0 + 1 + nx + nx*ny,
                    n0 + 0 + nx + nx*ny,
                ]
                elems.append(conn)
    return pts, jnp.array(elems, dtype=jnp.int32)

# ----------------------------- Small-strain linear elasticity element stiffness (constant material) for one hexa element

def element_stiffness_hex(coords_element, E=1e5, nu=0.3):
    # coords_element: (8,3)
    # We'll use single-point (center) integration for this demo.
    xi = eta = zeta = 0.5  # center in [0,1]
    N = tensor_basis_3d(xi, eta, zeta)
    dNxi, dNeta, dNzeta = tensor_basis_3d_derivs(xi, eta, zeta)
    # build Jacobian
    dN_param = jnp.stack([dNxi, dNeta, dNzeta], axis=1)  # (8,3)
    J = dN_param.T @ coords_element  # (3,3)
    detJ = jnp.linalg.det(J)
    invJ = jnp.linalg.inv(J)
    # gradients in physical space
    grads = dN_param @ invJ.T  # (8,3)
    # build B matrix (6 x 24)
    B = jnp.zeros((6, 24))
    for a in range(8):
        gx, gy, gz = grads[a]
        idx = 3*a
        B = B.at[0, idx+0].set(gx)
        B = B.at[1, idx+1].set(gy)
        B = B.at[2, idx+2].set(gz)
        B = B.at[3, idx+0].set(gy)
        B = B.at[3, idx+1].set(gx)
        B = B.at[4, idx+1].set(gz)
        B = B.at[4, idx+2].set(gy)
        B = B.at[5, idx+0].set(gz)
        B = B.at[5, idx+2].set(gx)
    # elasticity matrix D (isotropic)
    lam = (E*nu)/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    D = jnp.array([
        [lam+2*mu, lam, lam, 0,0,0],
        [lam, lam+2*mu, lam,0,0,0],
        [lam, lam, lam+2*mu,0,0,0],
        [0,0,0,mu,0,0],
        [0,0,0,0,mu,0],
        [0,0,0,0,0,mu]
    ])
    ke = B.T @ D @ B * detJ
    return ke

# vectorize element stiffness
vec_elem_k = jax.vmap(element_stiffness_hex, in_axes=(0,None,None))

# ----------------------------- Chord (1D truss) element stiffness

def chord_stiffness(p1, p2, A=1e-4, E=1e6):
    Lvec = p2 - p1
    L = jnp.linalg.norm(Lvec)
    if L < 1e-12:
        return jnp.zeros((6,6))
    n = Lvec / L
    k_local = (A*E / L) * (jnp.outer(n,n))
    # build 6x6 in 3D: [k -k; -k k]
    k = jnp.zeros((6,6))
    k = k.at[0:3,0:3].set(k_local)
    k = k.at[0:3,3:6].set(-k_local)
    k = k.at[3:6,0:3].set(-k_local)
    k = k.at[3:6,3:6].set(k_local)
    return k

# ----------------------------- Assembly (global)

def assemble_global(patch_pts, elems, E=1e5, nu=0.3):
    n_nodes = patch_pts.shape[0]
    dof = 3*n_nodes
    rows = []
    cols = []
    data = []
    for e_idx in range(elems.shape[0]):
        conn = elems[e_idx]
        coords_e = patch_pts[conn]
        ke = element_stiffness_hex(coords_e, E, nu)
        # assemble
        for a_local, a in enumerate(conn):
            for b_local, b in enumerate(conn):
                for i in range(3):
                    for j in range(3):
                        rows.append(3*a + i)
                        cols.append(3*b + j)
                        data.append(float(ke[3*a_local+i, 3*b_local+j]))
    K = coo_matrix((data, (rows, cols)), shape=(dof, dof)).tocsr()
    return K

# ----------------------------- Demo: one patch + one chord, penalty tie

def demo():
    # make a single patch 2x2x2 control grid -> single hexa element
    pts, elems = make_hexa_patch(2,2,2, origin=(0.0,0.0,0.0), size=(10.0,10.0,2.0))
    pts = jnp.array(pts)
    elems = jnp.array(elems)
    K = assemble_global(pts, elems, E=2e5, nu=0.3)
    n_nodes = pts.shape[0]
    dof = 3*n_nodes
    # chords: one chord connecting a point on top surface to a remote attachment
    # pick insertion point at param (xi,eta,zeta) = (0.5,0.5,1.0) -> top face center
    # evaluate shape functions at that param to get physical coordinate
    xi = eta = 0.5
    zeta = 1.0 - 1e-12
    N = tensor_basis_3d(xi, eta, zeta)
    # for degree-1 patch our control nodes are the physical nodes; evaluate
    # map uses same ordering as tensor_basis_3d
    insertion_pt = (N[:,None] * pts[elems[0]]).sum(axis=0)
    # chord other end (papillary muscle) somewhere below
    pap_pt = jnp.array([5.0, 5.0, -10.0])
    # create a new global system including chord DOFs
    # We'll attach chord by adding its stiffness into global K and adding penalty to tie chord node to insertion_pt
    # For simplicity create chord node as separate global node
    chord_node_idx = n_nodes
    # extend K to include chord node (3 dof)
    K_ext = coo_matrix( ([], ([],[])), shape=(dof+3, dof+3) ).tocsr()
    # copy existing K
    K_ext = K_ext.tolil()
    K_ext[:dof,:dof] = K
    # add chord stiffness between insertion_pt (as a free node) and pap_pt
    # But insertion_pt is not an existing grid node; so we model chord as between a new chord "anchor" node (attached via penalty to solid) and pap node
    # create pap node in global system
    pap_idx = chord_node_idx + 1
    # coordinates for chord nodes
    chord_coords = jnp.vstack([insertion_pt, pap_pt])
    k_ch = chord_stiffness(chord_coords[0], chord_coords[1], A=1e-3, E=1e6)
    # place k_ch into K_ext at indices [chord_node_idx, pap_idx]
    # expand K_ext's shape
    new_size = dof + 6
    K_big = coo_matrix(([], ([],[])), shape=(new_size, new_size)).tolil()
    K_big[:dof,:dof] = K_ext[:dof,:dof]
    # insert k_ch
    idx_map = [dof + i for i in range(6)]
    for i in range(6):
        for j in range(6):
            K_big[idx_map[i], idx_map[j]] += k_ch[i,j]
    # Penalty: tie chord first node DOFs (index dof..dof+2) to solid insertion physical point (which is a linear combination of element node DOFs)
    # enforce: u_chord_node - sum(N_a * u_solid_a) = 0 -> penalty term Kp = alpha * (T^T T)
    alpha = 1e8
    # T maps solid DOFs to constraint DOFs (3 x total_dof)
    # For simplicity compute T vector for each displacement direction
    # Build T_row for x DOF of constraint: entries at solid dofs corresponding to element nodes times N
    conn = elems[0]
    T = jnp.zeros(dof + 6)
    # entries for solid side
    for a_local, a in enumerate(conn):
        for k_dir in range(3):
            T = T.at[3*a + k_dir].set(float(-N[a_local]))
    # entries for chord DOFs: +1 at chord node x,y,z
    for k_dir in range(3):
        T = T.at[dof + k_dir].set(1.0)
    # add penalty contribution Kp = alpha * T^T T -> contributes to all pairs
    T = np.array(T)
    Kp = alpha * (T[:,None] @ T[None,:])
    # add Kp into K_big
    K_big = K_big.tocsr()
    K_big = K_big + coo_matrix(Kp)
    # Define force: apply downward force at papillary node in -z
    F = np.zeros(new_size)
    F[dof + 5] = -1e3  # apply at pap second node z DOF (indexing: dof+3..dof+5 are pap)
    # apply essential BC: fix bottom face z=0 nodes (for demo fix node 0)
    fixed_dofs = [2]  # fix z of node 0 (index 0 -> dof 2)
    # also fix rigid body modes by fixing x,y of node 0
    fixed_dofs += [0,1]
    free = np.setdiff1d(np.arange(new_size), fixed_dofs)
    K_ff = K_big[free][:,free]
    F_f = F[free]
    sol = spsolve(K_ff.tocsc(), F_f)
    U = np.zeros(new_size)
    U[free] = sol
    print("Solid node displacements:")
    for i in range(n_nodes):
        print(i, U[3*i:3*i+3])
    print("Chord node displacement:", U[dof:dof+3])

if __name__ == '__main__':
    demo()

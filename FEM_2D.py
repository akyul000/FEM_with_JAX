import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)


class Quadrature:
    def __init__(self, dim, p, f):
        self.dim = dim
        self.p = p
        self.f = f


    def rule(self):
        if self.p<= 1:
            self.n = 1
            self.x = [0.]
            self.w = [2.]
        elif self.p<= 3:
            self.n = 2
            self.x = [1/np.sqrt(3), -1/np.sqrt(3)]
            self.w = [1.,1.]
        elif self.p<= 5:
            self.n = 3
            self.x = [np.sqrt(0.6), 0, -np.sqrt(0.6)]
            self.w = [5/9, 8/9, 5/9]



    def integrate(self):
        self.rule()
        gauss_quadrature = 0

        if self.dim == 1:
            for i in range(self.n): 
                gauss_quadrature += self.w[i]*self.f(self.x[i])
            print(gauss_quadrature)


        elif self.dim == 2:
            for i in range(self.n):
                for j in range(self.n):
                    gauss_quadrature += self.w[i]* self.w[j]*self.f(self.x[i],self.x[j])
        

        else: 
            raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")
        return gauss_quadrature
        
def shape_functions_bilinear(xi, eta):
    N1 = 1 / 4 * (1 - xi) * (1 - eta)
    N2 = 1 / 4 * (1 + xi) * (1 - eta)
    N3 = 1 / 4 * (1 + xi) * (1 + eta)
    N4 = 1 / 4 * (1 - xi) * (1 + eta)
    return np.array([N1, N2, N3, N4])

# Wrap inputs in a single vector for differentiation
def shape_fn_wrapped(xi_eta):
    xi, eta = xi_eta
    return shape_functions_bilinear(xi, eta)


# Compute the Jacobian: each row is dN_i/d[xi, eta]
shape_fn_jacobian = jax.jacfwd(shape_fn_wrapped)  # or jax.jacrev



def integrand_modularized(xi, eta, physical_points, T):
    grads = shape_fn_jacobian(np.array([xi, eta]))  # (4, 2): rows = dN/dξ, dN/dη
    J =  physical_points.T @ grads
    physical_shape_grads = grads @ np.linalg.inv(J)
    dN_dx = physical_shape_grads[:, 0].reshape(-1, 1)  # (4,1)
    dN_dy = physical_shape_grads[:, 1].reshape(-1, 1)  # (4,1)
    integrand = T * (dN_dx @ dN_dx.T + dN_dy @ dN_dy.T) * np.linalg.det(J)
    return integrand







def make_integrand_wrapper(T, points_batch):
    """
    Returns a function f(xi, eta) that applies over all elements in the batch.
    X_batch, Y_batch: shape (n_elem, 4, 1)
    """
    def f(xi, eta):
        return jax.vmap(lambda physical_points: integrand_modularized(xi, eta, physical_points, T))(points_batch)
    return f  # f(xi, eta) → (n_elem, 4, 4)

def vmapped_integrand(points_batch, T):
    f = make_integrand_wrapper(T)
    
    def f_single(xi, eta):
        return jax.vmap(lambda physical_points: f(xi, eta, physical_points))(points_batch)
    
    return f_single


######################################### Meshing #########################################

## TODO: Test this
dim = 2
p = 1    
T = 2





Nx = 24
Ny = 6
NoE = Nx * Ny



NpE = 2
NoN_x = Nx + 1
NoN_y = Ny + 1
NoN = NoN_x * NoN_y
element_no = 0
element_nodal_coordinates = {}
element_nodal_numbers = {}
element_nodal_coords_batched = []
element_nodal_numbers_list = []


# 2D parameter grids
x_grid = np.linspace(0, 1, NoN_x)
y_grid = np.linspace(0, 1, NoN_y)



# 2D meshgrid
X, Y = np.meshgrid(x_grid, y_grid)

# Flatten and stack into (N, 2) shape
points = np.stack([X.ravel(), Y.ravel()], axis=-1)

import time

element_as_start = time.time()
for i in range(Ny):
    for j in range(Nx):
        lower_left = j + (Nx + 1) * i
        lower_right = j + (Nx + 1) * i + 1

        upper_left = j + (Nx + 1) * (i + 1)
        upper_right = j + (Nx + 1) * (i + 1) + 1
        element_nodes_global = np.array([lower_left, lower_right, upper_right, upper_left])
        element_nodal_coordinates[element_no] = points[element_nodes_global]
        element_nodal_numbers[element_no] = element_nodes_global
        element_nodal_numbers_list.append(element_nodes_global)
        element_nodal_coords_batched.append(points[element_nodes_global])

        element_no += 1
element_as_end = time.time()
print(element_as_end-element_as_start)

element_nodal_numbers_ar = np.array(element_nodal_numbers_list)
element_nodal_coords_batched = np.array(element_nodal_coords_batched)
def find_boundary_nodes(nodes, x_val=1.0, tol=1e-8):
    # Boolean mask for nodes with x ≈ x_val
    mask = np.abs(nodes[:, 0] - x_val) < tol
    return np.where(mask)[0]  # Ind

# Set up to Dirichlet BCs
left_bc = 0
right_bc = 0.25
all_nodes = np.arange(0,NoN,1)
dirichlet_bounds = np.array([0,1])
inds_boundary = np.sort(np.array([find_boundary_nodes(points, x_val=i) for i in dirichlet_bounds]).reshape(-1))
free_inds = np.delete(all_nodes, inds_boundary)
##################################################################################
def assemble_global_stiffness_loop(NoN,NoE,element_nodal_coordinates):
    K = np.zeros((NoN,NoN))
    for m in range(NoE):
        element_node_numbers = element_nodal_numbers[m]
        K_e = ke_all[m]
        for i in range(len(element_node_numbers)):
            for j in range(len(element_node_numbers)):
                K_enrtry_updated = K[element_node_numbers[i],element_node_numbers[j]] + K_e[i,j]
                K = K.at[element_node_numbers[i],element_node_numbers[j]].set(K_enrtry_updated)
    return K

def assemble_global_stiffness(ke_all, element_nodal_numbers, NoN):
    """
    More memory-efficient assembly using index_add.
    
    Args:
        ke_all: (n_elements, 4, 4) array of element stiffness matrices
        element_nodal_numbers: dictionary mapping element numbers to global node numbers
        NoN: Number of nodes in the mesh
    """
    # Convert element_nodal_numbers to an array (n_elements, 4)
    elem_nodes = np.array([element_nodal_numbers[i] for i in range(len(element_nodal_numbers))])
    
    # Create row and column indices for all elements
    rows = np.repeat(elem_nodes[:, :, None], 4, axis=2).flatten()
    cols = np.repeat(elem_nodes[:, None, :], 4, axis=1).flatten()
    values = ke_all.reshape(-1)
    
    # Use index_add to assemble directly
    K = np.zeros((NoN, NoN))
    K = K.at[(rows, cols)].add(values)
    return K

from scipy.sparse import coo_matrix

def assemble_sparse_stiffness_matrix(ke_all, element_nodal_numbers_ar, NoN):
    """
    Assemble the global stiffness matrix in sparse COO format.

    Args:
        ke_all: (n_elements, 4, 4) stiffness matrices
        element_nodal_numbers_ar: (n_elements, 4) global node indices for each element
        NoN: number of nodes

    Returns:
        scipy.sparse.coo_matrix
    """
    n_elements = ke_all.shape[0]

    # Create index arrays for local-to-global mapping
    I_local = np.repeat(element_nodal_numbers_ar[:, :, None], 4, axis=2)  # (n_elem, 4, 4)
    J_local = np.repeat(element_nodal_numbers_ar[:, None, :], 4, axis=1)  # (n_elem, 4, 4)

    I_all = I_local.reshape(-1)
    J_all = J_local.reshape(-1)
    V_all = ke_all.reshape(-1)

    # Convert to numpy arrays (JAX arrays can't be used in SciPy)
    I_all_np = onp.array(I_all)
    J_all_np = onp.array(J_all)
    V_all_np = onp.array(V_all)

    K_sparse = coo_matrix((V_all_np, (I_all_np, J_all_np)), shape=(NoN, NoN))
    return K_sparse


# Build vmapped integrand function (xi, eta) → (batch_size,)
f = make_integrand_wrapper(T, element_nodal_coords_batched)
Q = Quadrature(dim=2, p=2, f=f)
ke_all = Q.integrate()  

K_sparse = assemble_sparse_stiffness_matrix(ke_all, element_nodal_numbers_ar, NoN)

import time
# Assemble global stiffness matrix using vmap
start_v = time.time()
K_vmapped = assemble_global_stiffness(ke_all, element_nodal_numbers, NoN)
end_v = time.time()

# kpu = K_vmapped[np.ix_(free_inds, free_inds)]
# kpp = K_vmapped[np.ix_(free_inds, inds_boundary)]

# fp = np.zeros_like(inds_boundary)

# up = np.array([0,0.25,0,0.25])

# u_ins = onp.linalg.solve(kpu, fp-kpp@up)

# u_comb = onp.zeros(8)

# # for i in all_nodes:
# #     for free in free_inds:
# #         if i == free:
# #             u_comb[i] = 
# u_comb[0] = 0
# u_comb[1] = u_ins[0]
# u_comb[2] = u_ins[1]
# u_comb[3] = 0.25
# u_comb[4] = 0
# u_comb[5] = u_ins[2]
# u_comb[6] = u_ins[3]
# u_comb[7] = 0.25

def solve_dirichlet_system(K, free_inds, inds_boundary, up, f=None):
    """
    Solve Ku=f under Dirichlet BCs.

    Args:
        K: (NoN, NoN) global stiffness matrix
        free_inds: indices of free DOFs
        inds_boundary: indices of Dirichlet DOFs
        up: prescribed values at Dirichlet DOFs
        f: (NoN,) RHS vector. If None, assumed zero.

    Returns:
        u_comb: full solution vector (NoN,)
    """
    NoN = K.shape[0]
    if f is None:
        f = np.zeros(NoN)
    
    # Extract submatrices
    Kuu = K[np.ix_(free_inds, free_inds)]
    Kup = K[np.ix_(free_inds, inds_boundary)]

    # Extract RHS
    fu = f[free_inds]

    # Solve system
    rhs = fu - Kup @ up
    uu = onp.linalg.solve(Kuu, rhs)

    # Combine solution
    u_comb = np.zeros(NoN)
    u_comb = u_comb.at[free_inds].set(uu)
    u_comb = u_comb.at[inds_boundary].set(up)
    return u_comb

def get_boundary_node_indices(points, x_vals, tol=1e-8):
    """
    Find indices of nodes that lie on specified x-coordinate values.
    """
    boundary_inds = [np.where(np.abs(points[:, 0] - val) < tol)[0] for val in x_vals]
    return np.sort(np.concatenate(boundary_inds))


dirichlet_values = {0.0: 0.0, 1.0: 0.25}
inds_boundary = get_boundary_node_indices(points, list(dirichlet_values.keys()))
free_inds = np.setdiff1d(np.arange(NoN), inds_boundary)
up = np.array([dirichlet_values[points[i, 0].item()] for i in inds_boundary])

u_comb = solve_dirichlet_system(K_vmapped, free_inds, inds_boundary, up)




# print(f"solution is {u_ins}")

# start_l = time.time()
# K = assemble_global_stiffness_loop(NoN, NoE, element_nodal_coordinates)
# end_l = time.time()


# print(np.linalg.norm(K_vmapped-K))
print(f"Time for the vmapped assembly is {end_v-start_v}")
# print(f"Time for the looped assembly is {end_l-start_l}")


plt.figure()
plt.spy(K_vmapped, markersize=2)
plt.show()


# print(K_sparse)
print(u_comb)





deformed_points = points[:,0] + u_comb

plt.figure()
plt.plot(points[:,0],points[:,1],"ro")
plt.plot(deformed_points,points[:,1],"go")

plt.show()
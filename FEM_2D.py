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

Nx = 80
Ny = 60
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

# Build vmapped integrand function (xi, eta) → (batch_size,)
f = make_integrand_wrapper(T, element_nodal_coords_batched)
Q = Quadrature(dim=2, p=2, f=f)
ke_all = Q.integrate()  


import time
# Assemble global stiffness matrix using vmap
start_v = time.time()
K_vmapped = assemble_global_stiffness(ke_all, element_nodal_numbers, NoN)
end_v = time.time()

# start_l = time.time()
# K = assemble_global_stiffness_loop(NoN, NoE, element_nodal_coordinates)
# end_l = time.time()


# print(np.linalg.norm(K_vmapped-K))
print(f"Time for the vmapped assembly is {end_v-start_v}")
# print(f"Time for the looped assembly is {end_l-start_l}")


plt.figure()
plt.spy(K_vmapped, markersize=2)
plt.show()

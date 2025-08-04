import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
import jax
from scipy.sparse import coo_matrix
jax.config.update("jax_enable_x64", True)

# Quadrature class (same as before)
class Quadrature:
    def __init__(self, dim, p, f):
        self.dim = dim
        self.p = p
        self.f = f

    def rule(self):
        if self.p <= 1:
            self.n = 1
            self.x = [0.]
            self.w = [2.]
        elif self.p <= 3:
            self.n = 2
            self.x = [1/np.sqrt(3), -1/np.sqrt(3)]
            self.w = [1.,1.]
        elif self.p <= 5:
            self.n = 3
            self.x = [np.sqrt(0.6), 0, -np.sqrt(0.6)]
            self.w = [5/9, 8/9, 5/9]

    def integrate(self):
        self.rule()
        result = 0.0

        if self.dim == 1:
            for i in range(self.n): 
                result += self.w[i]*self.f(self.x[i])
        elif self.dim == 2:
            for i in range(self.n):
                for j in range(self.n):
                    result += self.w[i]*self.w[j]*self.f(self.x[i], self.x[j])
        else: 
            raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")
        return result

# Shape functions
def shape_functions_bilinear(xi, eta):
    N1 = 1/4 * (1 - xi) * (1 - eta)
    N2 = 1/4 * (1 + xi) * (1 - eta)
    N3 = 1/4 * (1 + xi) * (1 + eta)
    N4 = 1/4 * (1 - xi) * (1 + eta)
    return np.array([N1, N2, N3, N4])

def shape_fn_wrapped(xi_eta):
    xi, eta = xi_eta
    return shape_functions_bilinear(xi, eta)

shape_fn_jacobian = jax.jacfwd(shape_fn_wrapped)

# Elasticity integrand
def integrand_elasticity(xi, eta, physical_points, E, nu):
    # Compute shape function gradients
    grads = shape_fn_jacobian(np.array([xi, eta]))  # (4, 2)
    J = physical_points.T @ grads  # Jacobian
    dN_dX = grads @ np.linalg.inv(J)  # (4,2) shape function gradients in physical coords
    
    # Strain-displacement matrix B (3x8 for plane stress)
    B = np.zeros((3, 8))
    for i in range(4):
        B = B.at[0, 2*i].set(dN_dX[i, 0])
        B = B.at[1, 2*i+1].set(dN_dX[i, 1])
        B = B.at[2, 2*i].set(dN_dX[i, 1])
        B = B.at[2, 2*i+1].set(dN_dX[i, 0])
    
    # Elasticity matrix D (plane stress)
    factor = E/(1-nu**2)
    D = factor * np.array([[1, nu, 0],
                         [nu, 1, 0],
                         [0, 0, (1-nu)/2]])
    
    return (B.T @ D @ B) * np.linalg.det(J)

def make_elasticity_integrand(E, nu, points_batch):
    def f(xi, eta):
        return jax.vmap(lambda pts: integrand_elasticity(xi, eta, pts, E, nu))(points_batch)
    return f

# Mesh generation (same as before)
Nx, Ny = 4, 3
NoE = Nx * Ny
NoN = (Nx + 1) * (Ny + 1)

x_grid = np.linspace(0, 1, Nx + 1)
y_grid = np.linspace(0, 1, Ny + 1)
X, Y = np.meshgrid(x_grid, y_grid)
points = np.stack([X.ravel(), Y.ravel()], axis=-1)

# Element connectivity
element_nodal_numbers_list = []
element_nodal_coords_batched = []
for i in range(Ny):
    for j in range(Nx):
        lower_left = j + (Nx + 1) * i
        lower_right = j + (Nx + 1) * i + 1
        upper_left = j + (Nx + 1) * (i + 1)
        upper_right = j + (Nx + 1) * (i + 1) + 1
        element_nodes = np.array([lower_left, lower_right, upper_right, upper_left])
        element_nodal_numbers_list.append(element_nodes)
        element_nodal_coords_batched.append(points[element_nodes])

element_nodal_numbers_ar = np.array(element_nodal_numbers_list)
element_nodal_coords_batched = np.array(element_nodal_coords_batched)

# Material properties
E = 1000.0  # Young's modulus
nu = 0.3    # Poisson's ratio
import time
# Assembly
f = make_elasticity_integrand(E, nu, element_nodal_coords_batched)
Q = Quadrature(dim=2, p=2, f=f)
start_time = time.time()
ke_all = Q.integrate()
end_time = time.time()
print(f"Elment stiffness batched time: {end_time - start_time:.4f} seconds")

def assemble_sparse_stiffness_matrix(ke_all, element_nodal_numbers_ar, NoN):
    n_elements = ke_all.shape[0]
    n_nodes_per_element = 4
    dof_per_node = 2
    
    # Create global DOF indices (each node has 2 DOFs: ux, uy)
    I = []
    J = []
    V = []
    
    for elem in range(n_elements):
        for i in range(n_nodes_per_element):
            for j in range(n_nodes_per_element):
                # Global node numbers
                ni = element_nodal_numbers_ar[elem, i]
                nj = element_nodal_numbers_ar[elem, j]
                
                # Add all 4 entries (ux-ux, ux-uy, uy-ux, uy-uy)
                for di in range(dof_per_node):
                    for dj in range(dof_per_node):
                        I.append(ni * dof_per_node + di)
                        J.append(nj * dof_per_node + dj)
                        V.append(ke_all[elem, i*dof_per_node + di, j*dof_per_node + dj])
    
    K_sparse = coo_matrix((V, (I, J)), shape=(NoN*dof_per_node, NoN*dof_per_node))
    return K_sparse
start_time = time.time()
K_sparse = assemble_sparse_stiffness_matrix(ke_all, element_nodal_numbers_ar, NoN)
K = K_sparse.tocsr()
end_time = time.time()
print(f"Assembly time: {end_time - start_time:.4f} seconds")

# Boundary conditions
def find_boundary_nodes(points, x_val=None, y_val=None, tol=1e-8):
    if x_val is not None:
        return np.where(np.abs(points[:, 0] - x_val) < tol)[0]
    elif y_val is not None:
        return np.where(np.abs(points[:, 1] - y_val) < tol)[0]
    else:
        return np.array([])

# Fix left edge (both x and y), apply x-displacement to right edge
left_nodes = find_boundary_nodes(points, x_val=0)
right_nodes = find_boundary_nodes(points, x_val=1.0)
bottom_nodes = find_boundary_nodes(points, y_val=0)
top_nodes = find_boundary_nodes(points, y_val=1.0)

# Dirichlet BCs: 0 = fixed, 1 = free
bc_dofs = []
bc_values = []

# Fix left edge completely (both x and y)
for node in left_nodes:
    bc_dofs.extend([2*node, 2*node+1])  # ux and uy
    bc_values.extend([0.0, 0.0])

# Apply x-displacement to right edge
for node in right_nodes:
    bc_dofs.append(2*node)  # ux only
    bc_values.append(0.1)   # x-displacement

# Fix bottom edge in y-direction only
for node in bottom_nodes:
    if node not in left_nodes:  # Don't double-count left corner
        bc_dofs.append(2*node+1)  # uy only
        bc_values.append(0.0)

# Convert BC lists to JAX arrays
bc_dofs_array = np.array(bc_dofs, dtype=np.int32)
bc_values_array = np.array(bc_values, dtype=np.float64)

all_dofs = np.arange(2*NoN)
free_dofs = np.setdiff1d(all_dofs, bc_dofs_array)
# Solve
from scipy.sparse.linalg import spsolve

K_free = K[free_dofs, :][:, free_dofs]
F = np.zeros(2*NoN)

# Apply boundary conditions
U = np.zeros(2*NoN)
U = U.at[bc_dofs_array].set(bc_values_array)
F_modified = F - K @ U

# Solve for free DOFs
U_free = spsolve(K_free, F_modified[free_dofs])
U = U.at[free_dofs].set(U_free)  # Another immutable update

# Extract displacements
u_x = U[::2]
u_y = U[1::2]

# Corrected plotting section
plt.figure(figsize=(15, 5))

# Original mesh
plt.subplot(131)
for elem in element_nodal_numbers_ar:
    # Convert indices to array explicitly
    idx = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    x_coords = points[elem[idx], 0]
    y_coords = points[elem[idx], 1]
    plt.plot(x_coords, y_coords, 'b-')
plt.title('Original Mesh')
plt.gca().set_aspect('equal')

# Deformed mesh (scaled for visualization)
scale = 5.0
deformed = points + scale * np.column_stack([u_x, u_y])
plt.subplot(132)
for elem in element_nodal_numbers_ar:
    idx = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    x_def = deformed[elem[idx], 0]
    y_def = deformed[elem[idx], 1]
    plt.plot(x_def, y_def, 'r-')
plt.title(f'Deformed Mesh (Scale: {scale}x)')
plt.gca().set_aspect('equal')

# Displacement fields
plt.subplot(133)
plt.quiver(points[:,0], points[:,1], u_x, u_y, scale=0.5)
plt.title('Displacement Vectors')
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()
# Contour plots
plt.figure(figsize=(15, 5))

plt.subplot(131)
tricontour = plt.tricontourf(points[:,0], points[:,1], u_x, levels=20)
plt.colorbar(tricontour, label='X-displacement')
plt.title('X-displacement Contour')
plt.gca().set_aspect('equal')

plt.subplot(132)
tricontour = plt.tricontourf(points[:,0], points[:,1], u_y, levels=20)
plt.colorbar(tricontour, label='Y-displacement')
plt.title('Y-displacement Contour')
plt.gca().set_aspect('equal')

plt.subplot(133)
tricontour = plt.tricontourf(points[:,0], points[:,1], np.sqrt(u_x**2 + u_y**2), levels=20)
plt.colorbar(tricontour, label='Displacement Magnitude')
plt.title('Displacement Magnitude')
plt.gca().set_aspect('equal')

plt.tight_layout()
plt.show()
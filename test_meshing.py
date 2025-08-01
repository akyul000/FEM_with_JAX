# import jax.numpy as np
# import numpy as onp
# import matplotlib.pyplot as plt

# Nx = 4
# Ny = 3

# NPE = 4

# # for i in range(Nx*Ny):
# #     lower_left = i
# #     lower_right = i + 1
# #     upper_left = i + 

# lower = np.arange(1,16,1)
# upper = lower + 5
# print(upper)
# print(lower)

# connectivity = {}
# for i in range(14):
#     print("*****")
#     print(upper[i:i+2])
#     print(lower[i:i+2])
#     print("*****")
#     connectivity[f"el_{i+1}"] = [lower[i:i+2],upper[i:i+2]]

import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

def generate_structured_quad_mesh_connectivity(Nx, Ny):
    """
    Generate connectivity for a structured quadrilateral mesh.
    
    Parameters:
    -----------
    Nx : int
        Number of elements along x-direction.
    Ny : int
        Number of elements along y-direction.

    Returns:
    --------
    connectivity : dict
        Dictionary of element connectivity with element IDs as keys.
        Each value is a list of 4 node indices [ll, lr, ur, ul] (counter-clockwise).
    """
    connectivity = {}
    num_nodes_x = Nx + 1
    element_id = 1

    for j in range(Ny):
        for i in range(Nx):
            ll = j * num_nodes_x + i          # lower left
            lr = j * num_nodes_x + (i + 1)    # lower right
            ul = (j + 1) * num_nodes_x + i    # upper left
            ur = (j + 1) * num_nodes_x + (i + 1)  # upper right
            
            # Store counter-clockwise node indices
            connectivity[f'el_{element_id}'] = [ll, lr, ur, ul]
            element_id += 1

    return connectivity

# Example usage
Nx, Ny = 4, 3
conn = generate_structured_quad_mesh_connectivity(Nx, Ny)

# Print for verification
for k, v in conn.items():
    print(f"{k}: {v}")

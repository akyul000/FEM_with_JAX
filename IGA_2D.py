import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from jax import jit
import jax

jax.config.update("jax_enable_x64", True)
import general_gauss_quadrature as ggq
import NURBS_helper_functions as NURBS 

# Parameters
p = 2
q = 2
P_x = 6
P_y = 6

Lx = 10.0
Ly = 10.0

F_N = -5
num_xi_array_for_post_processing = 50

cps_x = np.linspace(0, Lx, P_x)
cps_y = np.linspace(0, Ly, P_y)

# 2D meshgrid
X, Y = np.meshgrid(cps_x, cps_y,indexing="ij")

# Flatten and stack into (N, 2) shape
cps = np.stack([X.ravel(), Y.ravel()], axis=-1)


knot_x = NURBS.get_open_uniform_knot_vector(P_x, p)
knot_y = NURBS.get_open_uniform_knot_vector(P_y, q)


def integrand_modularized(xi, eta, x_knot,y_knot, p, q,cps):
    span_x_i = NURBS.find_span(x_knot, p, xi)
    span_y_i = NURBS.find_span(y_knot, q, eta)



    N_i_xi, dN_i_xi = NURBS.bspline_basis_and_derivatives(x_knot, p, span_x_i, xi)
    N_i_eta, dN_i_eta = NURBS.bspline_basis_and_derivatives(y_knot, q, span_y_i, eta)



    cps_idx_x = NURBS.get_control_point_indices(span_x_i, p)
    cps_idx_y = NURBS.get_control_point_indices(span_y_i, q)



    N_i = np.array(N_i).reshape(-1, 1)
    dN_i = np.array(dN_i).reshape(-1, 1)

    point = onp.zeros(3)
    dx_dxi = onp.zeros(3)
    dx_deta = onp.zeros(3)

    grad_u_param = onp.zeros((2,2))

    for a in range(p + 1):
        for b in range(q + 1):
 
            # Geometry control point
            cp = cps[cps_idx_x[a], cps_idx_y[b]]
       
            # Basis value and directional derivatives
            R = N_i_xi[a] * N_i_eta[b]
            dR_dxi = dN_i_xi[a] * N_i_eta[b] 
            dR_deta = N_i_xi[a] * dN_i_eta[b] 
   


    
  

            # Compute mapping
            point += R * cp
            dx_dxi += dR_dxi * cp
            dx_deta += dR_deta * cp
 
    
    # Construct the Jacobian matrix
    J = onp.column_stack((dx_dxi, dx_deta, dx_dzeta))  # shape: (3, 3)
    grad_u_physical = grad_u_param @ onp.linalg.inv(J)

    return point, grad_u_physical



    grads = shape_fn_jacobian(np.array([xi, eta]))  # (4, 2): rows = dN/dξ, dN/dη
    J =  cps.T @ grads
    physical_shape_grads = grads @ np.linalg.inv(J)
    dN_dx = physical_shape_grads[:, 0].reshape(-1, 1)  # (4,1)
    dN_dy = physical_shape_grads[:, 1].reshape(-1, 1)  # (4,1)
    integrand = T * (dN_dx @ dN_dx.T + dN_dy @ dN_dy.T) * np.linalg.det(J)
    return integrand


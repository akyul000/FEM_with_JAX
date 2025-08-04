


import jax.numpy as np
import numpy as onp
import jax
import functools as fctls
import meshio
from itertools import product

from vtk import VTK_HEXAHEDRON
import pyvista as pv




def find_span(knot, p, xi):
    """
    Finds the knot span index for a given parameter value in a B-spline or NURBS knot vector.
    Parameters:
        knot (list or array-like): The knot vector, a non-decreasing sequence of parameter values.
        p (int): The degree of the B-spline or NURBS basis functions.
        xi (float): The parameter value for which to find the knot span.
    Returns:
        int: The index of the knot span in which xi lies, such that knot[span] <= xi < knot[span+1].
              If xi is equal to the last knot value, returns the last valid span index.
    Notes:
        - Assumes that the knot vector is valid and non-decreasing.
        - The function uses a binary search for efficiency.
    """
    
    n = len(knot) - p - 1
    if xi == knot[n]:
        return n - 1
    low = p
    high = n
    mid = (low + high) // 2
    while xi < knot[mid] or xi >= knot[mid + 1]:
        if xi < knot[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def bspline_basis(knot, p, span, xi):
    """
    Computes the non-zero B-spline basis functions at a given parametric coordinate.
    Parameters
    ----------
    knot : array_like
        The knot vector, a non-decreasing sequence of parameter values.
    p : int
        The degree of the B-spline basis functions.
    span : int
        The knot span index such that knot[span] <= xi < knot[span+1].
    xi : float
        The parametric coordinate at which to evaluate the basis functions.
    Returns
    -------
    N : ndarray
        Array of length (p+1) containing the values of the non-zero B-spline basis functions at xi.
    """
    
    N = onp.zeros(p+1)
    left = onp.zeros(p+1)
    right = onp.zeros(p+1)
    N[0] = 1.0

    for j in range(1, p+1):
        left[j] = xi - knot[span + 1 - j]
        right[j] = knot[span + j] - xi
        saved = 0.0
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N

def get_control_point_indices(span, p):
    """
    Returns the indices of the control points that influence a given span in a B-spline or NURBS curve.
    Parameters:
        span (int): The index of the current knot span.
        p (int): The degree of the B-spline or NURBS curve.
    Returns:
        numpy.ndarray: An array of control point indices from (span - p) to (span), inclusive.
    """

    return onp.arange(span - p, span + 1)



def bspline_basis_and_derivatives(knots, degree, span, xi):
    """
    Compute the non-zero B-spline basis functions and their first derivatives at a given parameter value.
    Parameters
    ----------
    knots : array_like
        The knot vector of the B-spline.
    degree : int
        The degree of the B-spline basis functions (p).
    span : int
        The knot span index such that knots[span] <= xi < knots[span+1].
    xi : float
        The parameter value at which to evaluate the basis functions and their derivatives.
    Returns
    -------
    N : ndarray, shape (degree+1,)
        The values of the non-zero B-spline basis functions at xi.
    dN : ndarray, shape (degree+1,)
        The values of the first derivatives of the non-zero B-spline basis functions at xi.
    Notes
    -----
    This function computes only the first derivatives of the B-spline basis functions.
    """
    p = degree
    N = onp.zeros((p+1, p+1))  # N[i][j] is the j-th derivative of the i-th basis function
    ndu = onp.zeros((p+1, p+1))
    a = onp.zeros((2, p+1))
    left = onp.zeros(p+1)
    right = onp.zeros(p+1)

    ndu[0, 0] = 1.0

    for j in range(1, p+1):
        left[j] = xi - knots[span + 1 - j]
        right[j] = knots[span + j] - xi
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(p+1):
        N[j, 0] = ndu[j, p]

    # Derivatives
    for r in range(p+1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0
        for k in range(1, 2):  # only 1st derivative
            d = 0.0
            rk = r - k
            pk = p - k
            if rk >= 0:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            for j in range(1, k + 1):
                if rk + j >= 0 and rk + j <= pk:
                    a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                    d += a[s2, j] * ndu[rk + j, pk]
            N[r, k] = d
            s1, s2 = s2, s1

    # Multiply by factorial
    dN = onp.zeros(p+1)
    for i in range(p+1):
        dN[i] = N[i, 1] * p

    return N[:, 0], dN

from scipy.linalg import solve
import matplotlib.pyplot as plt

# --- Parameters ---
L = 1.0
E = 1.0
f_func = lambda x: 1.0  # Body force
traction = 0.0

# B-spline setup
p = 2
num_elem = 5
num_ctrlpts = num_elem + p
knot_vector = [0] * (p + 1) + list(onp.linspace(0, 1, num_elem + 1)[1:-1]) + [1] * (p + 1)
knot_vector = onp.array(knot_vector, dtype=float)

# Control point locations (uniform)
ctrl_pts = onp.linspace(0, L, num_ctrlpts)

# Quadrature points
quad_order = p + 1
quad_points_ref, quad_weights_ref = onp.polynomial.legendre.leggauss(quad_order)

# Global matrices
K = onp.zeros((num_ctrlpts, num_ctrlpts))
F = onp.zeros(num_ctrlpts)

# Loop over knot spans (elements)
unique_knots = onp.unique(knot_vector)
for e in range(len(unique_knots) - 1):
    u_start = unique_knots[e]
    u_end = unique_knots[e + 1]
    if onp.isclose(u_end - u_start, 0):
        continue

    # Map quadrature points to current element
    for q in range(quad_order):
        xi_hat = quad_points_ref[q]
        w_hat = quad_weights_ref[q]
        xi = ((u_end - u_start) * xi_hat + (u_end + u_start)) / 2.0
        weight = w_hat * (u_end - u_start) / 2.0

        span = find_span(knot_vector, p, xi)
        N, dN_dxi = bspline_basis_and_derivatives(knot_vector, p, span, xi)
        ctrl_inds = get_control_point_indices(span, p)

        # Compute dx/dxi = Jacobian
        dx_dxi = onp.dot(dN_dxi, ctrl_pts[ctrl_inds])
        jacobian = dx_dxi

        dN_dx = dN_dxi / jacobian
        x_physical = onp.dot(N, ctrl_pts[ctrl_inds])

        for i in range(p + 1):
            a = ctrl_inds[i]
            F[a] += f_func(x_physical) * N[i] * weight
            for j in range(p + 1):
                b = ctrl_inds[j]
                K[a, b] += E * dN_dx[i] * dN_dx[j] * weight

# Neumann BC
span = find_span(knot_vector, p, 1.0)
N = bspline_basis(knot_vector, p, span, 1.0)
ctrl_inds = get_control_point_indices(span, p)
for i in range(p + 1):
    F[ctrl_inds[i]] += N[i] * traction

# Dirichlet BC at x=0 (u=0)
u_d = 0.0
fixed_dof = 0
free_dof = onp.delete(onp.arange(num_ctrlpts), fixed_dof)

K_mod = K[onp.ix_(free_dof, free_dof)]
F_mod = F[free_dof] - K[onp.ix_(free_dof, [fixed_dof])].flatten() * u_d

# Solve
u = onp.zeros(num_ctrlpts)
u[free_dof] = solve(K_mod, F_mod)
u[fixed_dof] = u_d

# Plot
x_vals = onp.linspace(0, 1, 200)
u_vals = []
for xi in x_vals:
    span = find_span(knot_vector, p, xi)
    N = bspline_basis(knot_vector, p, span, xi)
    ctrl_inds = get_control_point_indices(span, p)
    u_vals.append(onp.dot(N, u[ctrl_inds]))

# plt.plot(x_vals, u_vals, label="IGA solution")
# plt.xlabel("x")
# plt.ylabel("u(x)")
# plt.title("1D Elasticity with IGA (B-splines)")
# plt.grid(True)
# plt.legend()
# plt.show()

u_exact = -0.5 * x_vals**2 + x_vals

plt.plot(x_vals, u_vals, label="IGA solution")
plt.plot(x_vals, u_exact, '--', label="Analytical solution")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Elasticity: IGA vs Analytical")
plt.grid(True)
plt.legend()
plt.show()

# Optionally, plot the error
plt.plot(x_vals, onp.abs(onp.array(u_vals) - u_exact), label="|Error|")
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Pointwise Error: IGA vs Analytical")
plt.grid(True)
plt.legend()
plt.show()

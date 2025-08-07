import numpy as onp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
import meshio


def get_open_uniform_knot_vector(num_cp, p):
    """
    Generates an open uniform knot vector for B-splines or NURBS.

    Parameters
    ----------
    num_cp : int
        Number of control points.
    p : int
        Degree of the B-spline or NURBS basis functions.

    Returns
    -------
    knot_vector : numpy.ndarray
        The open uniform knot vector of length (num_cp + p + 1).

    Notes
    -----
    - The open uniform knot vector starts with (p+1) zeros and ends with (p+1) ones.
    - The interior knots are uniformly spaced between 0 and 1.
    - Used for constructing B-spline/NURBS basis functions with open uniform parameterization.
    """
    # Generates an open uniform knot vector for B-splines/NURBS
    n = num_cp - 1
    m = p + n + 1
    len_knot_vector = m + 1

    knot_vector = onp.zeros(len_knot_vector)
    for i in range(p+1):
        knot_vector[-(i+1)] = 1
    remaining_len = len_knot_vector - 2 * (p)
    mid = onp.linspace(0,1,remaining_len)
    knot_vector[p:-(p)] = mid
    return knot_vector

def get_periodic_knot_vector(num_cp, p):
    """
    Generates a periodic knot vector for NURBS (Non-Uniform Rational B-Splines) based on the number of control points and the degree of the curve.
    Parameters
    ----------
    num_cp : int
        The number of control points for the NURBS curve.
    p : int
        The degree of the NURBS curve.
    Returns
    -------
    numpy.ndarray
        A 1D array representing the periodic knot vector, with length equal to (num_cp + p + 1).
    Notes
    -----
    - The periodic knot vector is constructed as a simple range of integers from 0 to (num_cp + p).
    - This function assumes a uniform spacing for the knots, which is typical for periodic NURBS curves.
    - The knot vector is essential for defining the parameterization of the NURBS curve.
    Examples
    --------
    >>> get_periodic_knot_vector(5, 3)
    array([0, 1, 2, 3, 4, 5, 6, 7])
    """
    n = num_cp - 1
    m = p + n + 1
    len_knot_vector = m + 1
    return onp.arange(0,len_knot_vector,1)



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

def evaluate_bspline_surface(knot_xi, knot_eta, control_points, xi, eta, p, q, **kwargs):
    """
    Evaluates a B-spline surface at the given parametric coordinates (xi, eta).
    Parameters:
        knot_xi (array-like): Knot vector in the xi direction.
        knot_eta (array-like): Knot vector in the eta direction.
        control_points (ndarray): Array of control points with shape (n_xi, n_eta, 3).
        xi (float): Parametric coordinate in the xi direction.
        eta (float): Parametric coordinate in the eta direction.
        p (int): Degree of the B-spline in the xi direction.
        q (int): Degree of the B-spline in the eta direction.
        **kwargs: Additional arrays of the same shape as control_points to be evaluated at (xi, eta). n 
    Returns:
        tuple:
            - point (ndarray): The evaluated point on the B-spline volume (shape: (3,)).
            - points_additional (ndarray): Evaluated values for each additional array passed via kwargs (shape: (len(kwargs), 3)).
    """
    span_xi = find_span(knot_xi, p, xi)
    span_eta = find_span(knot_eta, q, eta)
    
    
    basis_xi = bspline_basis(knot_xi, p, span_xi, xi)
    basis_eta = bspline_basis(knot_eta, q, span_eta, eta)
    
    
    cp_idx_xi = get_control_point_indices(span_xi, p)
    cp_idx_eta = get_control_point_indices(span_eta, q)
    
    
    point = onp.zeros(3)
    points_additional = onp.zeros((len(kwargs), 3))

    for a in range(p + 1):
        for b in range(q + 1):
            weight = basis_xi[a] * basis_eta[b]
            point += weight * control_points[cp_idx_xi[a], cp_idx_eta[b]]
            for idx, key in enumerate(kwargs):
                points_additional[idx] += weight * kwargs[key][cp_idx_xi[a], cp_idx_eta[b]]
    return point, points_additional


def get_design_matrix_A(n_points, theta, phi, p, q, u_knots, v_knots):
    """
    Constructs the design matrix A for least squares fitting of a B-spline/NURBS surface to a set of data points.

    Parameters
    ----------
    n_points : int
        Number of data points in the point cloud.
    theta : array_like
        Array of parametric coordinates in the u-direction (typically shape: (n_points,)).
    phi : array_like
        Array of parametric coordinates in the v-direction (typically shape: (n_points,)).
    p : int
        Degree of the B-spline basis functions in the u-direction.
    q : int
        Degree of the B-spline basis functions in the v-direction.
    u_knots : array_like
        Knot vector in the u-direction (length: n_ctrl_p + p + 1).
    v_knots : array_like
        Knot vector in the v-direction (length: v_n_ctrl_p + q + 1).

    Returns
    -------
    S : ndarray
        The design matrix of shape (n_points, (u_n_ctrl_p - 1) * v_n_ctrl_p), where each row corresponds to a data point
        and each column corresponds to a control point. The entries are products of B-spline basis functions evaluated
        at the parametric coordinates (theta, phi) for each data point.

    Notes
    -----
    - The matrix S is constructed such that S[i, j] gives the influence of control point j on data point i.
    - Handles periodicity in the u-direction by combining the first and last basis functions.
    - Used for least squares fitting to solve for control points that best approximate the input point cloud.
    """
    P_u = len(u_knots) - p - 1
    P_v = len(v_knots) - q - 1
    S = onp.zeros((n_points, (P_u - 1) * P_v))
    # Precompute basis values
    N_u = onp.zeros((n_points, P_u))
    N_v = onp.zeros((n_points, P_v))
    for k in range(n_points):
        xi = theta[k]
        eta = phi[k]

        span_u = find_span(u_knots, p, xi)
        span_v = find_span(v_knots, q, eta)

        N_u_k = bspline_basis(u_knots, p, span_u, xi)
        N_v_k = bspline_basis(v_knots, q, span_v, eta)

        # Fill the correct span indices
        for i in range(p+1):
            N_u[k, span_u - p + i] = N_u_k[i]
        for j in range(q+1):
            N_v[k, span_v - q + j] = N_v_k[j]
    # Fill the matrix S
    for i in range(P_u - 1):
        for j in range(P_v):
            col_index = i + (P_u - 1) * j
   
            if i == 0:
                S[:, col_index] = N_u[:, 0] * N_v[:, j] + N_u[:, P_u - 1] * N_v[:, j]
            else:
                S[:, col_index] = N_u[:, i] * N_v[:, j]
    return S

def get_CARDIAX_control_point_indexing(C, P_u, P_v):
    """
    Rearranges the flat control points array from the least squares solution into the CARDIAX surface control point grid.

    Parameters
    ----------
    C : ndarray
        Flat array of control points from the least squares solution, shape ((P_u-1)*P_v, 3).
        Each row corresponds to a control point in 3D space (x, y, z).
    P_u : int
        Number of control points in the u-direction (including periodic wrap).
    P_v : int
        Number of control points in the v-direction.

    Returns
    -------
    C_pt : ndarray
        3D array of control points arranged for CARDIAX surface, shape (P_u, P_v, 3).
        The control points are ordered such that the periodicity in the u-direction is handled
        by combining the first and last control points.

    Notes
    -----
    - The input control points C are ordered as produced by the least squares solution, which
      does not include the periodic wrap in the u-direction.
    - The output C_pt includes the periodic wrap, with the last control point in the u-direction
      being a duplicate of the first to ensure periodicity.
    - This function is specific to the CARDIAX surface fitting convention.
    """
    fig = plt.figure(figsize=(24,12))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_pt,y_pt,z_pt,c='k', label='mv points')
    ax.scatter(C[:,0],C[:,1],C[:,2], s = 1, label='NURBS control points')
    ax.scatter(C[0:P_u-1,0],C[0:P_u-1,1],C[0:P_u-1,2],c='y', s = 6, label='free-edge')
    ax.scatter(C[(P_u-1) * (P_v-1):(P_u-1)*P_v,0],C[(P_u-1) * (P_v-1):(P_u-1)*P_v,1],C[(P_u-1) * (P_v-1):(P_u-1)*P_v,2],c='r', s = 2, label='annulus')
    ax.legend()
    plt.show()
    C_pt = onp.zeros((P_u*P_v,3))
    for i in range(0,P_u):
        for j in range(0,P_v):
            if i == P_u -1:
                C_pt[P_u -1 + P_u * j,:] = C[(P_u - 1) * j]
            else:
                C_pt[i + P_u * j,:] = C[i + (P_u - 1) * j]
    return C_pt.reshape((P_u,P_v,3),order="F")



def get_least_square_solution(point_cloud, A, u_knot, v_knot, p, q):
    """
    Solves the least squares problem to fit a B-spline/NURBS surface to a given point cloud.

    Parameters
    ----------
    point_cloud : ndarray
        Array of data points to fit, shape (num_points, 3).
        Each row is a 3D point (x, y, z).
    A : ndarray
        Design matrix of shape (num_points, (P_u-1)*P_v), where each row corresponds to a data point
        and each column corresponds to a control point.
    u_knot : array_like
        Knot vector in the u-direction.
    v_knot : array_like
        Knot vector in the v-direction.
    p : int
        Degree of the B-spline basis functions in the u-direction.
    q : int
        Degree of the B-spline basis functions in the v-direction.

    Returns
    -------
    dict
        Dictionary containing:
        - 'knot_vector_u': Knot vector in the u-direction.
        - 'knot_vector_v': Knot vector in the v-direction.
        - 'control_points': 3D array of fitted control points, shape (P_u, P_v, 3).
        - 'degree_u': Degree in the u-direction.
        - 'degree_v': Degree in the v-direction.

    Notes
    -----
    - Uses scipy.sparse.linalg.lsqr to solve the least squares problem for each coordinate dimension.
    - The control points are rearranged to match the CARDIAX surface convention, handling periodicity in the u-direction.
    - The output can be used to evaluate the fitted surface or for further processing.
    """
    P_u = len(u_knot) - p - 1
    P_v = len(v_knot) - q - 1
    num_cps = (P_u-1) * P_v
    control_points = onp.zeros((num_cps, 3))
    
    for dim in range(3):
        b = point_cloud[:, dim]
        solution = lsqr(A, b)[0]
        control_points[:, dim] = solution

    # ST = onp.transpose(A)
    # STS = onp.matmul(ST,A)
    # inv_STS = onp.linalg.inv(STS)
    # STP = onp.matmul(ST,point_cloud)
    # control_points = onp.matmul(inv_STS,STP)
    control_points = get_CARDIAX_control_point_indexing(control_points, P_u, P_v)


    
    return {
        'knot_vector_u': u_knot,
        'knot_vector_v': v_knot,
        'control_points': control_points,
        'degree_u': p,
        'degree_v': q
    }


def get_fitting_error(num_pt, point_cloud, u_knot, v_knot, control_points, u, v, p, q):
    """
    Computes the fitting error between the original point cloud and the B-spline surface approximation.

    Parameters
    ----------
    num_pt : int
        Number of points in the point cloud.
    point_cloud : numpy.ndarray
        Array of shape (num_pt, 3) containing the original data points.
    u_knot : numpy.ndarray
        Knot vector in the u-direction.
    v_knot : numpy.ndarray
        Knot vector in the v-direction.
    control_points : numpy.ndarray
        Array of shape (P_u, P_v, 3) containing the fitted control points.
    u : numpy.ndarray
        Array of parametric coordinates in the u-direction for each point (shape: (num_pt,)).
    v : numpy.ndarray
        Array of parametric coordinates in the v-direction for each point (shape: (num_pt,)).
    p : int
        Degree of the B-spline basis functions in the u-direction.
    q : int
        Degree of the B-spline basis functions in the v-direction.

    Returns
    -------
    None
        Prints the maximum, minimum, and mean fitting errors to the console.

    Notes
    -----
    - For each point in the point cloud, the function evaluates the B-spline surface at the corresponding (u, v)
      parametric coordinates and computes the Euclidean distance to the original point.
    - The error array contains the fitting error for each point.
    - Useful for assessing the quality of the surface fit.
    """
    error = onp.zeros(num_pt)
    for i in range(0,num_pt):
        surface_val, _ = evaluate_bspline_surface(u_knot, v_knot, control_points,u[i], v[i],p,q)
        # print(surface_val)
        error[i] = onp.linalg.norm(point_cloud[i,:] - surface_val)
        

    # # scipy.io.savemat('error.mat', {'error':error})

    print('maximum error = ', onp.max(error))
    print('minimum error = ', onp.min(error))
    print('mean error = ', onp.mean(error))


def plot_control_mesh(cps):
    # plot the new control points
    #%matplotlib widget

    if cps.ndim == 4:
    

        n_u, n_v, n_w, _ = cps.shape

        fig = plt.figure(figsize=(18,9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot control mesh lines and collect dummy handles for legend
        u_line, = ax.plot([], [], [], color='blue', label='circumferential-u-direction')
        v_line, = ax.plot([], [], [], color='green', label='radial-v-direction')
        w_line, = ax.plot([], [], [], color='red', label='thickness-w-direction')

        # Plot lines along u-direction
        for v in range(n_v):
            for w in range(n_w):
                curve = cps[:, v, w, :]
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='blue')

        # Plot lines along v-direction
        for u in range(n_u):
            for w in range(n_w):
                curve = cps[u, :, w, :]
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='green')

        # Plot lines along w-direction
        for u in range(n_u):
            for v in range(n_v):
                curve = cps[u, v, :, :]
                ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='red', linewidth=4)

            # Plot control points
        ax.scatter(cps[..., 0], cps[..., 1], cps[..., 2], color='black', s=25)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D B-spline Solid Control Mesh")

        # Add legend
        ax.legend(handles=[u_line, v_line, w_line])
    else:
        n_u, n_v, _ = cps.shape

        fig = plt.figure(figsize=(18,9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot control mesh lines and collect dummy handles for legend
        u_line, = ax.plot([], [], [], color='blue', label='circumferential-u-direction')
        v_line, = ax.plot([], [], [], color='green', label='radial-v-direction')


        # Plot lines along u-direction
        for v in range(n_v):
            
            curve = cps[:, v, :]
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='blue')

        # Plot lines along v-direction
        for u in range(n_u):
    
            curve = cps[u, :, :]
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color='green')




        # Plot control points
        ax.scatter(cps[..., 0], cps[..., 1], cps[..., 2], color='black', s=25)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("3D B-spline Surface Control Mesh")

        # Add legend
        ax.legend(handles=[u_line, v_line])

    plt.tight_layout()
    plt.show()


def evaluate_bspline_volume(knot_xi, knot_eta, knot_zeta, control_points, xi, eta, zeta, p, q, r):
    """Computes the physical coordinates of given parametric space coordinates  
    (xi, eta, zeta) -> (x, y, z) """
    span_xi = find_span(knot_xi, p, xi)
    span_eta = find_span(knot_eta, q, eta)
    span_zeta = find_span(knot_zeta, r, zeta)
    
    basis_xi = bspline_basis(knot_xi, p, span_xi, xi)
    basis_eta = bspline_basis(knot_eta, q, span_eta, eta)
    basis_zeta = bspline_basis(knot_zeta, r, span_zeta, zeta)
    
    cp_idx_xi = get_control_point_indices(span_xi, p)
    cp_idx_eta = get_control_point_indices(span_eta, q)
    cp_idx_zeta = get_control_point_indices(span_zeta, r)
    
    point = onp.zeros(3)
    for a in range(p + 1):
        for b in range(q + 1):
            for c in range(r + 1):
                weight = basis_xi[a] * basis_eta[b] * basis_zeta[c]
                point += weight * control_points[cp_idx_xi[a], cp_idx_eta[b], cp_idx_zeta[c]]
    return point

def evaluate_bspline_solid_uniform_at_parametric_grid(Nx,Ny,Nz,knot_xi,knot_eta,knot_zeta,cps,p,q,r):
    xi_vals = onp.linspace(knot_xi[p], knot_xi[-(p+1)], Nx)
    eta_vals = onp.linspace(0, 1, Ny)
    zeta_vals = onp.linspace(0, 1, Nz)

    # Prepare arrays
    points_undeformed = []
    point_indices = {}  # Maps (i,j,k) to global index
    index = 0

    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            for k, zeta in enumerate(zeta_vals):
                # Geometry position
                point_undeformed = evaluate_bspline_volume(knot_xi=knot_xi,
                                                    knot_eta=knot_eta,
                                                    knot_zeta=knot_zeta,
                                                    control_points=cps,
                                                    xi=xi, eta=eta, zeta=zeta,
                                                    p=p, q=q, r=r)
            
        

                points_undeformed.append(point_undeformed)
                point_indices[(i, j, k)] = index
                index += 1


    points_undeformed = onp.array(points_undeformed)
    return point_indices, points_undeformed


def extrude_cps(cps, knot_u, knot_v, knot_w, p, q, r, thickness,control_mesh=True, unclamped_knot_vector = False, smooth_normals=False, **kwargs):
    """
    Given a B-spline surface cps (n_u, n_v, 3), extrudes along surface normals
    using basis derivatives into a volume of control points (n_u, n_v, n_w, 3)
    """

    if cps.ndim == 4:
        raise ValueError(f"Expected a surface (3D array: n_u x n_v x dim), but got a solid-like control point shape: {cps.shape}")
    elif cps.ndim != 3:
        raise ValueError(f"Invalid cps shape: {cps.shape}. Expected 3D control points array (n_u x n_v x dim).")
    

    n_u, n_v, _ = cps.shape
    n_w = len(knot_w) - r - 1
    delta_layer = thickness / (n_w - 1)

    # Output volume CPs
    cps_volume = onp.zeros((n_u, n_v, n_w, 3))

    # Greville abscissae
    if not unclamped_knot_vector:
        greville_u = compute_greville_abscissae(knot_u, p)
    else:
        greville_u = onp.linspace(knot_u[p], knot_u[-p-1], cps.shape[0])

    greville_v = compute_greville_abscissae(knot_v, q)
   

    # Compute normal per control point
    normals = onp.zeros((n_u, n_v, 3))

    for i_u, u in enumerate(greville_u):
        span_u = find_span(knot_u, p, u)
        N_u, dN_u = bspline_basis_and_derivatives(knot_u, p, span_u, u)
        u_indices = get_control_point_indices(span_u, p)

        for i_v, v in enumerate(greville_v):
            span_v = find_span(knot_v, q, v)
            N_v, dN_v = bspline_basis_and_derivatives(knot_v, q, span_v, v)
            v_indices = get_control_point_indices(span_v, q)

            Su = onp.zeros(3)
            Sv = onp.zeros(3)

            for a in range(p+1):
                for b in range(q+1):
                    cp = cps[u_indices[a], v_indices[b]]
                    Su += dN_u[a] * N_v[b] * cp
                    Sv += N_u[a] * dN_v[b] * cp

            n = onp.cross(Su, Sv)
            norm = onp.linalg.norm(n)
            if norm > 1e-8:
                n = n / norm
            else:
                # n = onp.array([0.0, 0.0, 1.0])  # Fallback
                n = onp.mean(normals[max(0, i_u-1):i_u+1, max(0, i_v-1):i_v+1], axis=(0, 1))
                n = n / (onp.linalg.norm(n) + 1e-8)
            normals[i_u, i_v] = n
    if smooth_normals:
        from scipy.ndimage import gaussian_filter
        normals = gaussian_filter(normals, sigma=1.0, mode='nearest')
        normals = normals / onp.linalg.norm(normals, axis=-1, keepdims=True)
    

    # Build extruded volume
    for k in range(n_w):
        cps_volume[:, :, k] = cps + delta_layer * k * normals

    if unclamped_knot_vector:
        for i in range(p):
            cps_volume[-(p-i)] = cps_volume[i]


    if control_mesh:
        plot_control_mesh(cps_volume)

    print(f"MV surface geometry extruded in w direction with {n_w-1} layers (number of {n_w} control points)...")


    return cps_volume



def compute_greville_abscissae(knots, degree):
    """
    Computes Greville abscissae for B-spline basis functions.

    Parameters:
    - knots: list or array of knot vector
    - degree: degree of the B-spline basis functions (p)

    Returns:
    - greville_pts: list of Greville abscissae
    """
    n_basis = len(knots) - degree - 1  # number of basis functions / control points
    greville_pts = [
        sum(knots[i+1 : i+1+degree]) / degree
        for i in range(n_basis)
    ]
    return greville_pts


def create_hex_elements(Nx, Ny, Nz, point_indices):
    cells = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            for k in range(Nz - 1):
                n0 = point_indices[(i, j, k)]
                n1 = point_indices[(i + 1, j, k)]
                n2 = point_indices[(i + 1, j + 1, k)]
                n3 = point_indices[(i, j + 1, k)]
                n4 = point_indices[(i, j, k + 1)]
                n5 = point_indices[(i + 1, j, k + 1)]
                n6 = point_indices[(i + 1, j + 1, k + 1)]
                n7 = point_indices[(i, j + 1, k + 1)]
                cells.append([n0, n1, n2, n3, n4, n5, n6, n7])
    return cells

def get_expanded_control_points_degenerate_array(cps, p):
    """
    Expands a given array of control points by duplicating the first `p` control points at the end of the array.
    This function is typically used in spline or surface modeling where degenerate control points are required
    at the boundaries to ensure proper evaluation of the surface or curve. The expanded array has `p` additional
    control points, which are copies of the first `p` control points, appended to the end.
    Parameters
    ----------
    cps : numpy.ndarray
        The input array of control points with shape (N, M, 3), where N is the number of control points,
        M is the number of control point sets (e.g., for a grid), and 3 corresponds to the spatial dimensions (x, y, z).
    p : int
        The number of degenerate control points to append. This is typically the degree of the spline or surface.
    Returns
    -------
    numpy.ndarray
        The expanded array of control points with shape (N + p, M, 3), where the last `p` control points are
        duplicates of the first `p` control points.
    Notes
    -----
    - The function assumes that the input array `cps` is at least two-dimensional and the last dimension is of size 3.
    - The degenerate control points are required for certain algorithms in geometric modeling, such as B-splines,
      to handle boundary conditions.
    """
    cps_exp = onp.zeros((cps.shape[0] + p, cps.shape[1], 3))
    cps_exp[:-p,:,:] = cps
    for i in range(p):
        cps_exp[-(p-i)] = cps_exp[i]

    return cps_exp

def visualize_NURBS_surface(cps, u_knot, v_knot, p ,q, n_samples=100):
    """
    Visualizes a NURBS (Non-Uniform Rational B-Spline) surface using matplotlib.
    Parameters
    ----------
    cps : ndarray
        A 3D numpy array of control points with shape (n_ctrl_pts_u, n_ctrl_pts_v, 3),
        where each control point is represented by its (x, y, z) coordinates.
    u_knot : ndarray
        The knot vector in the u-direction (parameter direction 1).
    v_knot : ndarray
        The knot vector in the v-direction (parameter direction 2).
    p : int
        The degree of the B-spline basis functions in the u-direction.
    q : int
        The degree of the B-spline basis functions in the v-direction.
    n_samples : int, optional
        Number of samples to evaluate along each parametric direction (default is 100).
    Returns
    -------
    None
        This function displays a 3D plot of the NURBS surface using matplotlib.
    Notes
    -----
    - The function creates a uniform grid in the parametric (u, v) space, evaluates the
        B-spline surface at each grid point, and plots the resulting surface.
    - The surface evaluation relies on the `evaluate_bspline_surface` function, which
        should compute the surface point given the knot vectors, control points, degrees,
        and parametric coordinates.
    - The plot uses the 'coolwarm' colormap and displays the surface with edge coloring.
    - The axes are labeled and the plot is titled for clarity.
    Example
    -------
    >>> visualize_NURBS_surface(cps, u_knot, v_knot, p, q, n_samples=100)
    Displays a 3D plot of the NURBS surface defined by the given control points and knot vectors.
    """

    # Create parametric grid (uniform)
    u_vals = onp.linspace(u_knot[p], u_knot[-p-1], n_samples)
    v_vals = onp.linspace(v_knot[q], v_knot[-q-1], n_samples)
    u_grid, v_grid = onp.meshgrid(u_vals, v_vals)

    # Initialize arrays for surface
    X = onp.zeros_like(u_grid)
    Y = onp.zeros_like(v_grid)
    Z = onp.zeros_like(u_grid)

    # Evaluate surface at each grid point
    for i in range(n_samples):
        for j in range(n_samples):
            xi = u_grid[i, j]
            eta = v_grid[i, j]
            point, _ = evaluate_bspline_surface(u_knot, v_knot, cps, xi, eta, p, q)
            X[i, j], Y[i, j], Z[i, j] = point

    # Plot the surface using matplotlib
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='k', alpha=0.8)
    ax.plot_surface(X, Y, Z, cmap='coolwarm',edgecolor='k')

    ax.set_title("B-Spline Surface")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()




def get_vtu_file_bspline_evaluated_saved_hex(Nx,Ny,Nz,knot_xi,knot_eta,knot_zeta,cps,p,q,r,output_name):
    point_indices, points_undeformed = evaluate_bspline_solid_uniform_at_parametric_grid(Nx,Ny,Nz,knot_xi,knot_eta,knot_zeta,cps,p,q,r)
    cells = create_hex_elements(Nx=Nx,Ny=Ny, Nz=Nz,point_indices=point_indices)
    # Create and write VTU
    cells = [("hexahedron", onp.array(cells))]
    mesh = meshio.Mesh(
        points=points_undeformed,
        cells=cells)

    mesh.write(output_name)
    print(f"VTU mesh solid geometry saved as '{output_name}'")


def evaluate_bspline_volume(knot_xi, knot_eta, knot_zeta, control_points, xi, eta, zeta, p, q, r,**kwargs):
    """
    Evaluates a B-spline volume at the given parametric coordinates (xi, eta, zeta).
    Parameters:
        knot_xi (array-like): Knot vector in the xi direction.
        knot_eta (array-like): Knot vector in the eta direction.
        knot_zeta (array-like): Knot vector in the zeta direction.
        control_points (ndarray): Array of control points with shape (n_xi, n_eta, n_zeta, 3).
        xi (float): Parametric coordinate in the xi direction.
        eta (float): Parametric coordinate in the eta direction.
        zeta (float): Parametric coordinate in the zeta direction.
        p (int): Degree of the B-spline in the xi direction.
        q (int): Degree of the B-spline in the eta direction.
        r (int): Degree of the B-spline in the zeta direction.
        **kwargs: Additional arrays of the same shape as control_points to be evaluated at (xi, eta, zeta).
    Returns:
        tuple:
            - point (ndarray): The evaluated point on the B-spline volume (shape: (3,)).
            - points_additional (ndarray): Evaluated values for each additional array passed via kwargs (shape: (len(kwargs), 3)).
    """
    span_xi = find_span(knot_xi, p, xi)
    span_eta = find_span(knot_eta, q, eta)
    span_zeta = find_span(knot_zeta, r, zeta)
    
    basis_xi = bspline_basis(knot_xi, p, span_xi, xi)
    basis_eta = bspline_basis(knot_eta, q, span_eta, eta)
    basis_zeta = bspline_basis(knot_zeta, r, span_zeta, zeta)
    
    cp_idx_xi = get_control_point_indices(span_xi, p)
    cp_idx_eta = get_control_point_indices(span_eta, q)
    cp_idx_zeta = get_control_point_indices(span_zeta, r)
    
    point = onp.zeros(3)
    points_additional = onp.zeros((len(kwargs), 3))

    for a in range(p + 1):
        for b in range(q + 1):
            for c in range(r + 1):
                weight = basis_xi[a] * basis_eta[b] * basis_zeta[c]
                point += weight * control_points[cp_idx_xi[a], cp_idx_eta[b], cp_idx_zeta[c]]
                for idx, key in enumerate(kwargs):
                    points_additional[idx] += weight * kwargs[key][cp_idx_xi[a], cp_idx_eta[b], cp_idx_zeta[c]]
    return point, points_additional

def get_u_grads_NURBS(knot_xi, knot_eta, knot_zeta, control_points, cp_sol, xi, eta, zeta, p, q, r):
    """
    Computes the value and physical gradient of a vector field defined by NURBS basis functions at a given parametric point.
    Parameters
    ----------
    knot_xi : array_like
        Knot vector in the xi direction.
    knot_eta : array_like
        Knot vector in the eta direction.
    knot_zeta : array_like
        Knot vector in the zeta direction.
    control_points : ndarray
        Array of control points defining the geometry, shape (n_xi, n_eta, n_zeta, 3).
    cp_sol : ndarray
        Array of solution values (e.g., displacement or field values) at control points, shape (n_xi, n_eta, n_zeta, 3).
    xi : float
        Parametric coordinate in the xi direction.
    eta : float
        Parametric coordinate in the eta direction.
    zeta : float
        Parametric coordinate in the zeta direction.
    p : int
        Degree of the NURBS basis in the xi direction.
    q : int
        Degree of the NURBS basis in the eta direction.
    r : int
        Degree of the NURBS basis in the zeta direction.
    Returns
    -------
    point : ndarray
        The value of the vector field at the given parametric point, shape (3,).
    grad_u_physical : ndarray
        The physical gradient of the vector field at the given parametric point, shape (3, 3).
    Notes
    -----
    - The function computes the parametric derivatives of the NURBS basis functions and maps them to the physical domain using the Jacobian of the geometry mapping.
    - The returned gradient is with respect to the physical coordinates.
    """
    # Find spans
    span_xi = find_span(knot_xi, p, xi)
    span_eta = find_span(knot_eta, q, eta)
    span_zeta = find_span(knot_zeta, r, zeta)
    
    # Basis functions and derivatives
    basis_xi, dbasis_xi = bspline_basis_and_derivatives(knot_xi, p, span_xi, xi)
    basis_eta, dbasis_eta = bspline_basis_and_derivatives(knot_eta, q, span_eta, eta)
    basis_zeta, dbasis_zeta = bspline_basis_and_derivatives(knot_zeta, r, span_zeta, zeta)
    
    # Control point index ranges
    cp_idx_xi = get_control_point_indices(span_xi, p)
    cp_idx_eta = get_control_point_indices(span_eta, q)
    cp_idx_zeta = get_control_point_indices(span_zeta, r)
    
    # Initialize output
    point = onp.zeros(3)
    dx_dxi = onp.zeros(3)
    dx_deta = onp.zeros(3)
    dx_dzeta = onp.zeros(3)

    # Parametric gradient of vector field: shape (3, 3)
    grad_u_param = onp.zeros((3, 3))
    
    for a in range(p + 1):
        for b in range(q + 1):
            for c in range(r + 1):
                # Geometry control point
                cp = control_points[cp_idx_xi[a], cp_idx_eta[b], cp_idx_zeta[c]]
                sol = cp_sol[cp_idx_xi[a], cp_idx_eta[b], cp_idx_zeta[c]] # u 
                # Basis value and directional derivatives
                R = basis_xi[a] * basis_eta[b] * basis_zeta[c]
                dR_dxi = dbasis_xi[a] * basis_eta[b] * basis_zeta[c]
                dR_deta = basis_xi[a] * dbasis_eta[b] * basis_zeta[c]
                dR_dzeta = basis_xi[a] * basis_eta[b] * dbasis_zeta[c]


                grad_u_param[:, 0] += dR_dxi * sol
                grad_u_param[:, 1] += dR_deta * sol
                grad_u_param[:, 2] += dR_dzeta * sol

                # Compute mapping
                point += R * sol
                dx_dxi += dR_dxi * cp
                dx_deta += dR_deta * cp
                dx_dzeta += dR_dzeta * cp
    
    # Construct the Jacobian matrix
    J = onp.column_stack((dx_dxi, dx_deta, dx_dzeta))  # shape: (3, 3)
    grad_u_physical = grad_u_param @ onp.linalg.inv(J)

    return point, grad_u_physical


def evaluate_geometry(knot_xi, knot_eta, knot_zeta, xi_vals, eta_vals, zeta_vals, p, q, r, cps, **kwargs):
    """
    Evaluates the geometry and additional fields at specified parametric points within a B-spline volume.
    Parameters
    ----------
    knot_xi : array-like
        Knot vector in the xi direction.
    knot_eta : array-like
        Knot vector in the eta direction.
    knot_zeta : array-like
        Knot vector in the zeta direction.
    xi_vals : array-like
        List or array of xi parametric coordinates to evaluate.
    eta_vals : array-like
        List or array of eta parametric coordinates to evaluate.
    zeta_vals : array-like
        List or array of zeta parametric coordinates to evaluate.
    p : int
        Degree of the B-spline in the xi direction.
    q : int
        Degree of the B-spline in the eta direction.
    r : int
        Degree of the B-spline in the zeta direction.
    cps : array-like
        Control points of the B-spline volume.
    **kwargs : dict, optional
        Additional field data to be evaluated and returned. Each key should correspond to a field name.
    Returns
    -------
    point_indices : dict
        Mapping from (i, j, k) indices in the parametric grid to global point indices.
    points : numpy.ndarray
        Array of evaluated geometry points at the specified parametric locations.
    additional_field_data : dict
        Dictionary mapping each additional field name (from kwargs) to an array of evaluated field values at the specified points.
    Notes
    -----
    This function relies on `evaluate_bspline_volume` to compute the geometry and additional fields at each parametric location.
    """

    # Prepare arrays
    points = []
    point_indices = {}  # Maps (i,j,k) to global index
    index = 0
    additional_field_data = {key: [] for key in kwargs}
    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            for k, zeta in enumerate(zeta_vals):
                # Geometry position
                point, additional_fields = evaluate_bspline_volume(knot_xi=knot_xi,
                                                    knot_eta=knot_eta,
                                                    knot_zeta=knot_zeta,
                                                    control_points=cps,
                                                    xi=xi, eta=eta, zeta=zeta,
                                                    p=p, q=q, r=r, **kwargs)


             
                points.append(point)
                point_indices[(i, j, k)] = index
                index += 1
                for idx, key in enumerate(kwargs):
                    additional_field_data[key].append(additional_fields[idx])
    points = onp.array(points)
    for key in additional_field_data:
        additional_field_data[key] = onp.array(additional_field_data[key])
    return point_indices, points, additional_field_data

def evaluate_physical_quantities(knot_xi, knot_eta, knot_zeta, xi_vals, eta_vals, zeta_vals, p, q, r, sol_cp_reshaped, undeformed_cp_reshaped, **kwargs):
    """
    Evaluates physical quantities at specified parametric points within a NURBS or B-spline volume.
    This function computes the deformed and undeformed positions, displacement gradients, deformation gradients,
    and any additional fields at a grid of parametric points defined by xi_vals, eta_vals, and zeta_vals.
    It uses the provided knot vectors, control points, and polynomial degrees to evaluate the geometry and solution fields.
    Parameters
    ----------
    knot_xi : array_like
        Knot vector in the xi direction.
    knot_eta : array_like
        Knot vector in the eta direction.
    knot_zeta : array_like
        Knot vector in the zeta direction.
    xi_vals : array_like
        Parametric coordinates in the xi direction at which to evaluate quantities.
    eta_vals : array_like
        Parametric coordinates in the eta direction at which to evaluate quantities.
    zeta_vals : array_like
        Parametric coordinates in the zeta direction at which to evaluate quantities.
    p : int
        Degree of the basis functions in the xi direction.
    q : int
        Degree of the basis functions in the eta direction.
    r : int
        Degree of the basis functions in the zeta direction.
    sol_cp_reshaped : array_like
        Solution control points (typically deformed positions or displacements), reshaped for evaluation.
    undeformed_cp_reshaped : array_like
        Undeformed geometry control points, reshaped for evaluation.
    **kwargs : dict
        Additional fields to be evaluated and returned. Each key should correspond to a field name.
    Returns
    -------
    u_grads : numpy.ndarray
        Array of displacement gradient tensors at each evaluation point.
    F : numpy.ndarray
        Array of deformation gradient tensors at each evaluation point.
    point_indices : dict
        Mapping from (i, j, k) grid indices to global point indices.
    points_sol : numpy.ndarray
        Array of deformed (solution) positions at each evaluation point.
    points_undeformed : numpy.ndarray
        Array of undeformed positions at each evaluation point.
    additional_field_data : dict
        Dictionary mapping each additional field name to an array of its evaluated values at each point.
    """

    # Prepare arrays
    points_sol = []
    points_undeformed = []
    point_indices = {}  # Maps (i,j,k) to global index
    index = 0
    u_grads = []
    F = []
    additional_field_data = {key: [] for key in kwargs}
    for i, xi in enumerate(xi_vals):
        for j, eta in enumerate(eta_vals):
            for k, zeta in enumerate(zeta_vals):
                # Geometry position
                point_undeformed, additional_fields = evaluate_bspline_volume(knot_xi=knot_xi,
                                                    knot_eta=knot_eta,
                                                    knot_zeta=knot_zeta,
                                                    control_points=undeformed_cp_reshaped,
                                                    xi=xi, eta=eta, zeta=zeta,
                                                    p=p, q=q, r=r, **kwargs)
                # Obtain physical value of control point displacement and u_grads
                point_deformed, u_grad = get_u_grads_NURBS(knot_xi=knot_xi,
                                                        knot_eta=knot_eta,
                                                        knot_zeta=knot_zeta,cp_sol=sol_cp_reshaped,control_points=undeformed_cp_reshaped,xi=xi, eta=eta, zeta=zeta,
                                                        p=p, q=q, r=r)
                F_val = u_grad + onp.eye(3)
                u_grads.append(u_grad)
                F.append(F_val)
                points_sol.append(point_deformed)
                points_undeformed.append(point_undeformed)
                point_indices[(i, j, k)] = index
                index += 1
                for idx, key in enumerate(kwargs):
                    additional_field_data[key].append(additional_fields[idx])

    u_grads = onp.array(u_grads)
    F = onp.array(F)
    points_sol = onp.array(points_sol)
    points_undeformed = onp.array(points_undeformed)
    for key in additional_field_data:
        additional_field_data[key] = onp.array(additional_field_data[key])
    return u_grads, F, point_indices, points_sol, points_undeformed, additional_field_data


# def evaluate_physical_quantities_vmapped(knot_xi, knot_eta, knot_zeta, Nx, Ny, Nz, p, q, r, sol_cp_reshaped, undeformed_cp_reshaped, compute_greville = False):
#     if not compute_greville:
#         xi_vals = onp.linspace(0, 1, Nx)
#         eta_vals = onp.linspace(0, 1, Ny)
#         zeta_vals = onp.linspace(0, 1, Nz)
#     else: 
#         xi_vals = compute_greville_abscissae(knots=knot_xi, degree=p)
#         eta_vals = compute_greville_abscissae(knots=knot_eta, degree=q)
#         zeta_vals = compute_greville_abscissae(knots=knot_zeta, degree=r)

#     parametric_points = np.array(list(product(xi_vals, eta_vals, zeta_vals)))  # shape (N_points, 3)

#     def eval_undeformed(point):
#         xi, eta, zeta = point
#         return evaluate_bspline_volume(knot_xi=knot_xi,
#                                         knot_eta=knot_eta,
#                                         knot_zeta=knot_zeta,
#                                         control_points=undeformed_cp_reshaped,
#                                         xi=xi, eta=eta, zeta=zeta,
#                                         p=p, q=q, r=r)

#     def eval_deformed_and_grad(point):
#         xi, eta, zeta = point
#         return get_u_grads_NURBS(knot_xi=knot_xi,
#                                 knot_eta=knot_eta,
#                                 knot_zeta=knot_zeta,cp_sol=sol_cp_reshaped,control_points=undeformed_cp_reshaped,xi=xi, eta=eta, zeta=zeta,
#                                 p=p, q=q, r=r)



#     v_eval_undeformed = jax.vmap(eval_undeformed)
#     v_eval_deformed_and_grad = jax.vmap(eval_deformed_and_grad)

#     # Apply
#     import time
#     st_time = time.time()
#     points_undeformed = v_eval_undeformed(parametric_points)  # (N, 3)
#     end_time = time.time()
#     time_first = end_time - st_time
#     print(f'Time is {time_first}')
#     points_sol, u_grads = v_eval_deformed_and_grad(parametric_points)  # (N, 3), (N, 3, 3)

import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
import jax

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
# shape_functions_grads_bilinear = jax.grad(shape_functions_bilinear)


X = np.array([0,1,1,0]).reshape(-1,1)
Y = np.array([0,0,1,1]).reshape(-1,1)

xi = 0.25
eta = 0.3
T = 2

shape_functions_bilinear(xi, eta)
grads = shape_fn_jacobian(np.array([xi, eta])) # Shape (4, 2): each row is [dN_i/dxi, dN_i/deta]
print(grads)


def integrand_modularized(xi, eta, X, Y, T):
    grads = shape_fn_jacobian(np.array([xi, eta]))  # (4, 2): rows = dN/dξ, dN/dη
    physical_points = np.concatenate([X, Y], axis=1).T
    J =  physical_points @ grads

    physical_shape_grads = grads @ np.linalg.inv(J)
    dN_dx = physical_shape_grads[:, 0].reshape(-1, 1)  # (4,1)
    dN_dy = physical_shape_grads[:, 1].reshape(-1, 1)  # (4,1)

   
    integrand = T * (dN_dx @ dN_dx.T + dN_dy @ dN_dy.T) * np.linalg.det(J)
    return integrand




## TODO: Test this
dim = 2
p = 1    







def make_integrand_wrapper(T, X_batch, Y_batch):
    """
    Returns a function f(xi, eta) that applies over all elements in the batch.
    X_batch, Y_batch: shape (n_elem, 4, 1)
    """
    def f(xi, eta):
        return jax.vmap(lambda X, Y: integrand_modularized(xi, eta, X, Y, T))(X_batch, Y_batch)
    return f  # f(xi, eta) → (n_elem, 4, 4)

def vmapped_integrand(X_batch, Y_batch, T):
    f = make_integrand_wrapper(T)
    
    def f_single(xi, eta):
        return jax.vmap(lambda X, Y: f(xi, eta, X, Y))(X_batch, Y_batch)
    
    return f_single

# ---- Test Case ----
# Original single-element shape: (4, 1)
X = np.array([0, 1, 1, 0]).reshape(-1, 1)
Y = np.array([0, 0, 1, 1]).reshape(-1, 1)

# Create batched input: 1000 elements with same geometry
X_batch = np.array(np.tile(X, (1000000, 1)).reshape(1000000, 4, 1))
Y_batch = np.array(np.tile(Y, (1000000, 1)).reshape(1000000, 4, 1))
T = 2  # Dummy scalar parameter

# Build vmapped integrand function (xi, eta) → (batch_size,)
f = make_integrand_wrapper(T, X_batch, Y_batch)
Q = Quadrature(dim=2, p=2, f=f)
ke_all = Q.integrate()  # shape (1000, 4, 4)

# Evaluate it at a single quadrature point

print("ke[0]:\n", ke_all[0])
print("Shape:", ke_all.shape)




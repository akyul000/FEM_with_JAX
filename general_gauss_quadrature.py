import jax.numpy as np

class Quadrature:
    def __init__(self, dim, p, f):
        self.dim = dim
        self.p = p
        self.f = f  # f should take mapped xi values

    def rule(self):
        if self.p <= 1:
            self.n = 1
            self.x = np.array([0.])
            self.w = np.array([2.])
        elif self.p <= 3:
            self.n = 2
            val = 1.0 / np.sqrt(3.0)
            self.x = np.array([-val, val])
            self.w = np.array([1.0, 1.0])
        elif self.p <= 5:
            val = np.sqrt(0.6)
            self.n = 3
            self.x = np.array([-val, 0.0, val])
            self.w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        else:
            raise NotImplementedError("Higher-order quadrature not implemented.")

    def integrate(self, a=0., b=1.):
        self.rule()
        result = 0.

        if self.dim == 1:
            for i in range(self.n):
                xi_hat = self.x[i]
                xi_phys = ((b - a) / 2) * xi_hat + (a + b) / 2
                weight = self.w[i] * (b - a) / 2
                result += weight * self.f(xi_phys)
        else:
            raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")

        return result
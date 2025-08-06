import jax.numpy as np

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



        elif self.dim == 2:
            for i in range(self.n):
                for j in range(self.n):
                    gauss_quadrature += self.w[i]* self.w[j]*self.f(self.x[i],self.x[j])
        

        else: 
            raise NotImplementedError(f"Integration for dim={self.dim} not implemented.")
        return gauss_quadrature
        
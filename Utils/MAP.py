import numpy as np
from scipy.special import iv

class MAP:
    
    def __init__(self, r, sigma2, mu0, sigma02, size):
        
        self.r = r
        self.sigma2 = sigma2
        self.mu0 = mu0
        self.sigma02 = sigma02
        self.size = size
        self.its = 10
    
    def bessd(self, x, limval=700.0):
        y = np.where(x > np.full(self.size, limval), 1, iv(1, x)/iv(0, x))
        return y
    
    def calculate_MAP(self):
        A = np.multiply(self.mu0, self.sigma2)
        B = np.multiply(self.r, self.sigma02)
        C = np.add(self.sigma2, self.sigma02)
        D = np.divide(self.r, self.sigma2)
        rho = self.r
        
        for i in range(self.its):
            b = self.bessd(np.multiply(rho, D))
            rho = np.divide(np.add(A, np.multiply(B, b)), C)
            rho = np.where(rho < np.zeros(self.size), 0.0, rho)
              
        return rho


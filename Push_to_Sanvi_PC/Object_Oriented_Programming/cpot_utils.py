#Importing libraries
import numpy as np

#Classes
class Hamiltonian:
    def __init__(self, alpha, Er, N,q):
        self.mass = 1  #mass of atom
        self.alpha = alpha #lattice depth
        self.N = N #
        self.dims = 2*N + 1 # dimension of hamiltonian matrix
        self.q = q #quasi momentum
        self.hbar_scaled = 1 #scaled planck constant
        self.k_l = 1 #scaled lattice wavenumber
        self.Er = (self.hbar_scaled**2)*(self.k_l**2)/(2*self.mass) #recoil energy

    def potential(self):
         
        return self.alpha*self.Er
    

    def H_builder(self):
        V_0 = self.potential()
        main_diag = np.zeros(self.dims)
        for i in range(-self.N, self.N+1):
            main_diag[i + self.N] = ((2*i + self.q) / (self.hbar_scaled * self.k_l))**2 * self.Er
        offset_diag = np.full(self.dims-1, V_0/4)
        H = np.zeros((self.dims,self.dims), dtype= float)
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], offset_diag)
        np.fill_diagonal(H[:,1:], offset_diag)
        return H, main_diag, offset_diag

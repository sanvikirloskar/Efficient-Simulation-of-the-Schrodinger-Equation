import numpy as np

#i dont know if i even need these
import matplotlib.pyplot as plt
import scipy as sp

#Building hamiltonian
def H_builder(alpha, N, q, hbar, k_l, m):
  dims = 2*N + 1
  E_r = (hbar**2)*(k_l**2)/(2*m)
  V_0 = alpha*E_r
  main_diag = np.zeros(dims)
  for i in range(-N,N+1):
      main_diag[i+N] = (((2*i+q)/(hbar*k_l))**2*E_r)
      offset_diag = np.full(dims-1, -V_0/4)
      H = np.zeros((dims,dims), dtype= float)
      np.fill_diagonal(H, main_diag)
      np.fill_diagonal(H[1:], offset_diag)
      np.fill_diagonal(H[:,1:], offset_diag)
  return H

#params
alpha = 1
N = 10
q = 0
hbar = 1
k_l  = 1
m = 1
unit_cell = np.pi / k_l

#initial state
eigenvalues, eigenvectors = np.linalg.eigh(H_builder(alpha, N, q, hbar, k_l, m))

l = np.arange(-N, N + 1)
c = eigenvectors[:, 0].astype(complex)
x_grid = np.linspace(-2 * unit_cell, 2 * unit_cell, 1000)

phase_basis = np.exp(1j * 2 * k_l * l[:, None] * x_grid[None, :])
u = np.sum(c[:, None] * phase_basis, axis=0)

#initial and target states
psi_initial = np.exp(1j * q * x_grid) * u
psi_initial = c.copy()
frac = 1/(np.sqrt(2))
psi_target = frac * (np.exp(-1j * 2 * k_l * x_grid) * u + np.exp(1j * 2 * k_l * x_grid) * u)

#----------NELDER MEAD ALGORITHM---------------------#
def NM(psi_initial):
   #Defining search spaces and other terms
    #N_w = 2*N + 1
    #N_c  = 20
     
   #-------------------CRAB CODE------------------# 
   #Defining basis
    N_basis = 10
    np.random.seed(0)
    random_freqs = np.random.uniform(0, 2*np.pi, N_basis)
    def control_field(t, coeffs):
        A = 0
        B = 0
        for k in range(N_basis):
            A += coeffs[k] * np.sin(random_freqs[k] * t)
            B += coeffs[k] * np.cos(random_freqs[k] * t)
        return A,B
    def evolve(coeffs):
        psi = psi_initial.copy()
        times = np.linspace(0, 5, 300)
        dt = times[1] - times[0]
        for t in times:
            A_t, B_t = control_field(t, coeffs)
            H = H_builder(alpha, N, q, hbar, k_l, m) + (A_t + B_t) * np.eye(2*N+1)
            U = sp.linalg.expm(-1j * H * dt)
            psi = U @ psi
        return psi
    psi_final = evolve(np.random.rand(N_basis))
    return psi_final
    
NM(psi_initial)
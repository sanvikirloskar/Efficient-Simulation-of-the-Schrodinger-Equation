import numpy as np
import matplotlib.pyplot as plt
#parameters
alpha = 10
N = 10
m = 1 
dims = 2*N + 1
hbar = 1
k_l = 1
q = 0
unit_cell = 2 * np.pi / k_l
band = 0
num_points = 2000
def H_builder(alpha, N, q, hbar, k_l, m):
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

hamiltonian = H_builder(alpha, N, q, hbar, k_l, m)
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

l = np.arange(-N, N+1)
c = eigenvectors[:, band].astype(complex)
x_grid = np.linspace(-unit_cell, unit_cell, num_points)

phase_basis = np.exp(1j * 2 * k_l * l[:, None] * x_grid[None, :])
u = np.sum(c[:, None] * phase_basis, axis=0)

psi = np.exp(1j * q * x_grid) * u
psi_real = np.real(psi)
psi_real /= np.max(np.abs(psi_real)) #rescaled for plotting
# psi_imag = np.imag(psi)
# psi_prob = np.abs(psi)**2

plt.plot(x_grid, psi_real)
plt.xlabel('x')
plt.ylabel(r'$\mathrm{Re}(\psi(x))$')
plt.show()


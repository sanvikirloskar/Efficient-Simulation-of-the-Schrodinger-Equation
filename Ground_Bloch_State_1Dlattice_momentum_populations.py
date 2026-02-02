import numpy as np
import matplotlib.pyplot as plt
#parameters
alpha = 10
N = 10
m = 1 
dims = 2*N + 1
hbar = 1
k_l = np.sqrt(2)
q = 0
unit_cell = np.pi / k_l
band = 0


num_points = 300
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
c = eigenvectors[:, band]
x_grid = np.linspace(-unit_cell, unit_cell, num_points)

phase_basis = np.exp(1j * 2 * k_l * l[:, None] * x_grid[None, :])
u = np.sum(c[:, None] * phase_basis, axis=0)

psi = np.exp(1j * q * x_grid) * u
psi_real = np.real(psi)
psi_imag = np.imag(psi)
psi_imag /= np.max(np.abs(psi_imag))
psi_real /= np.max(np.abs(psi_real))
idx = len(x_grid)//2   # x ≈ 0
if np.real(psi_real[idx]) < 0:
    psi_real *= -1
# psi_real /= np.max(np.abs(psi_real)) #rescaled for plotting
# psi_imag = np.imag(psi)
# psi_prob = np.abs(psi)**2

# plt.plot(x_grid, psi_real, label=r'$\mathrm{Re}(\psi(x))$')
# # plt.plot(x_grid, psi_imag, label=r'$\mathrm{Re}(\psi(x))$')
# plt.xlabel('x')
# plt.ylabel(r'$\psi(x)$')
# plt.show()

p_units = q / k_l + 2 * l   
population = np.abs(c)**2
print("Sum |c_l|^2 =", np.sum(population)) #check normalisation

plt.figure()
plt.bar(p_units, population)
plt.xlabel(r"Momentum ($\hbar k_L$)")
plt.ylabel("Relative population")
plt.xticks(np.arange(min(p_units), max(p_units)+1, 2))
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#parameters
alpha = 10
N = 10
m = 1
dims = 2*N + 1
hbar = 1
k_l = 1

def H_builder(alpha, N, q, hbar, k_l, m):
    E_r = (hbar**2)*(k_l**2)/(2*m)
    V_0 = alpha*E_r
    main_diag = np.zeros(dims)
    for i in range(-N,N+1):
        main_diag[i+N] = (((2*i+q)/hbar*k_l)**2*E_r)
    offset_diag = np.full(dims-1, V_0/4)
    H = np.zeros((dims,dims), dtype= float)
    np.fill_diagonal(H, main_diag)
    np.fill_diagonal(H[1:], offset_diag)
    np.fill_diagonal(H[:,1:], offset_diag)
    return H, main_diag, offset_diag

q_vals = np.linspace(-hbar*k_l, hbar*k_l, 500)
energy_bands = np.zeros((len(q_vals), 2*N+1))

for i, q in enumerate(q_vals):
    H, main_diag, offset_diag = H_builder(alpha, N, q, hbar, k_l, m)
    eigenvalues = sp.linalg.eigh_tridiagonal(main_diag, offset_diag)[0]
    energy_bands[i, :] = eigenvalues  

print(energy_bands)
#Plotting the energy bands
lines = []
for band in range(5):
    line, = plt.plot(q_vals, energy_bands[:, band], label=f'Band {band + 1}')
    lines.append(line)

plt.xlabel('q Value ($\hbar k$)')
plt.ylabel('Energy')

# Create a custom legend showing band labels with their respective colors
plt.legend(handles=lines, labels=[f'Band {i+1}' for i, line in enumerate(lines)])

plt.show()
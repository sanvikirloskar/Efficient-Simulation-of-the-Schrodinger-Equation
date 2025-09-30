import numpy as np
import matplotlib.pyplot as plt
#parameters
alpha = 10
N = 3
m = 1
dims = 2*N + 1
hbar = 1
k_l = 2 * np.pi / 829

def H_builder(alpha, N, q, hbar, k_l, m):
    E_r = (hbar**2)*(k_l**2)/(2*m)
    V_0 = alpha*E_r
    main_diag = np.zeros(dims)
    for i in range(-N,N):
        main_diag[i] = (((2*i+q)/hbar*k_l)**2*E_r)
    offset_diag = np.full(dims-1, V_0/4)
    H = np.zeros((dims,dims), dtype= float)
    np.fill_diagonal(H, main_diag)
    np.fill_diagonal(H[1:], offset_diag)
    np.fill_diagonal(H[:,1:], offset_diag)
    return H

q_vals = np.linspace(-hbar*k_l, hbar*k_l, 200)
energy_bands = np.zeros((len(q_vals), 2*N+1))

for i, q in enumerate(q_vals):
    H = H_builder(alpha, N, q, hbar, k_l, m)
    eigenvalues = np.linalg.eigh(H)[0]
    energy_bands[i, :] = eigenvalues  

print(energy_bands)
#Plotting the energy bands
for band in range(2 * N + 1):
    plt.plot(q_vals, energy_bands[:, band], label=f'Band {band + 1}')

plt.xlabel('q (Momentum)')
plt.ylabel('Energy')
plt.title('Energy Bands in the First Brillouin Zone')
plt.legend()
plt.show()


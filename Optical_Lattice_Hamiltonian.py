import numpy as np
#parameters
alpha = 10
N = 3
q = 1
dims = 2*N + 1
hbar = 1
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

def compute_eigenvectors(alpha, N, q, hbar, k_l, m):
    H = H_builder(alpha, N, q, hbar, k_l, m)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors

# Example usage:
# k_l and m need to be defined before calling this function.
# eigenvalues, eigenvectors = compute_eigenvectors(alpha, N, q, hbar, k_l, m)

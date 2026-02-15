import numpy as np
import matplotlib.pyplot as plt
from QuEvolutio.quevolutio.core.domain import QuantumConstants

class OLConstants(QuantumConstants):
    """
    Represents the constants of an optical lattice system in natural
    units through the QuantumConstants interface.

    Attributes
    ----------
    hbar : float
        The reduced Planck constant.
    mass : float
        The mass of the system.
    v0 : float
        The potential of the lattice
    kl : float
        The wavevector of the lattice
    alpha : float
        The lattice depth parameter
    q : float
        The quasimomentum
    band : int
        The band index
    N : int
    unit_cell : float
        The size of the unit cell.
    num_points : int
        The number of spatial discretisation points.
    """

    hbar: float = 1.0
    mass: float = 1.0
    kl: float = 1.0
    alpha: float = 10.0
    q: float = 0.0
    band: int = 0
    N: int = 10
    num_pts: int = 150
    @property
    def unit_cell(self):
        return np.pi / self.kl
    
    @property
    def dims(self):
        return 2 * self.N + 1

    @property
    def E_r(self):
        return ((self.hbar **2) * (self.kl **2)) / (2 * self.mass)

    @property
    def V0(self):
        return self.alpha * self.E_r 

class GroundBlochState():

    def __init__(self):
        self.c = OLConstants()
        self.H = self.H_builder()
        self.x_grid = np.linspace(-self.c.unit_cell, self.c.unit_cell, self.c.num_pts)
    def H_builder(self):
        main_diag = np.zeros(self.c.dims)
        for i in range(-self.c.N, self.c.N + 1):
            main_diag[i + self.c.N] = (((2*i + self.c.q) / (self.c.hbar * self.c.kl))**2 * self.c.E_r)
        offset_diag = np.full(self.c.dims-1, -self.c.V0 / 4)
        H = np.zeros((self.c.dims, self.c.dims), dtype= float)
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], offset_diag)
        np.fill_diagonal(H[:,1:], offset_diag)
        return H
    
    def generate_bloch_state(self, spacial_dims=1): # if you don't specify the dimensions it will automatically be 1d, but 2d can be called explicitly
        l = np.arange(-self.c.N, self.c.N +1)
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        c = eigenvectors[:, self.c.band]
        phase_basis = np.exp(1j * 2 * self.c.kl * l[:, None] * self.x_grid[None, :])
        u = np.sum(c[:, None] * phase_basis, axis=0)
        psi1d = np.exp(1j * self.c.q * self.x_grid) * u
        psi1d_real = np.real(psi1d)
        psi1d_real /= np.max(np.abs(psi1d_real))
        if spacial_dims == 1:
            return psi1d_real
        if spacial_dims == 2:
            psi2d = psi1d_real[:, None] * psi1d_real[None, :]
            return psi2d
    



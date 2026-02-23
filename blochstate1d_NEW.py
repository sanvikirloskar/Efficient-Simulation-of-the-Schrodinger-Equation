import numpy as np
import matplotlib.pyplot as plt
from quevolutio.core.domain import QuantumConstants
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
    kl : float
        The wavevector of the lattice.
    alpha : float
        The lattice depth parameter.
    q : float
        The quasimomentum.
    band : int
        The band index.
    N : int
        Truncated basis of fourier series for creating bloch states.
    num_points : int
        The number of spatial discretisation points.
    N_basis : int
        Number of basis functions in the CRAB shaking function.
    n_p : int
        Number of momentum states to consider in the momentum population calculation.
    unit_cell : float
        The size of the unit cell.
    lower_x_bound : float
        Lower bound of the position grid.
    upper_x_bound : float
        Upper bound of the position grid.
    dims: int
        Dimension of Hamiltonian matrix for creating bloch states.
    E_r : float
        Lattice recoil energy.
    v0 : float
        The potential of the lattice.
    """

    hbar: float = 1.0
    mass: float = 1.0
    kl: float = 1.0
    alpha: float = 10.0
    q: float = 0
    band: int = 0
    N: int = 10
    num_pts: int = 150
    N_basis: int = 10
    n_p = 11 # -10hbark to +10hbark
    N_individuals = 40 #This is A in the genetic algorithm part of Carries thesis (doubled like we did for CRAB)
    N_elements = 100
    @property
    def unit_cell(self):
        return np.pi / self.kl
    @property
    def lower_x_bound(self):
        return - self.unit_cell
    @property
    def upper_x_bound(self):
        return self.unit_cell
    @property
    def dims(self):
        return 2 * self.N + 1

    @property
    def E_r(self):
        return ((self.hbar **2) * (self.kl **2)) / (2 * self.mass)

    @property
    def v0(self):
        return self.alpha * self.E_r 
    
    @property
    #split state
    def desired_mom_pop(self):
            mom_pop_des = np.zeros(11) 
            mom_pop_des[4] = 0.5 
            mom_pop_des[6] = 0.5
            return mom_pop_des

class GroundBlochState():

    def __init__(self):
        self.c = OLConstants()
        self.H = self.H_builder()
        self.x_grid = np.linspace(self.c.lower_x_bound, self.c.upper_x_bound, self.c.num_pts)
    def H_builder(self):
        main_diag = np.zeros(self.c.dims)
        for i in range(-self.c.N, self.c.N + 1):
            main_diag[i + self.c.N] = (((2*i + self.c.q) / (self.c.hbar * self.c.kl))**2 * self.c.E_r)
        offset_diag = np.full(self.c.dims-1, -self.c.v0 / 4)
        H = np.zeros((self.c.dims, self.c.dims), dtype= float)
        np.fill_diagonal(H, main_diag)
        np.fill_diagonal(H[1:], offset_diag)
        np.fill_diagonal(H[:,1:], offset_diag)
        return H
    
    def eigenvalues(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        eigvals = eigenvalues
        eigvecs = eigenvectors
        return eigvals, eigvecs

    def generate_bloch_state(self, spacial_dims=1): # if you don't specify the dimensions it will automatically be 1d, but 2d can be called explicitly
        l = np.arange(-self.c.N, self.c.N +1)
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        c = eigenvectors[:, self.c.band]
        phase_basis = np.exp(1j * 2 * self.c.kl * l[:, None] * self.x_grid[None, :])
        u = np.sum(c[:, None] * phase_basis, axis=0)
        psi1d = np.exp(1j * self.c.q * self.x_grid) * u
        psi1d_real = np.real(psi1d)
        psi1d_real /= np.max(np.abs(psi1d_real))
        idx = len(self.x_grid)//2   # x ≈ 0
        if np.real(psi1d_real[idx]) < 0:
            psi1d_real *= -1
        if spacial_dims == 1:
            return psi1d_real
        if spacial_dims == 2:
            psi2d = psi1d_real[:, None] * psi1d_real[None, :]
            return psi2d
    



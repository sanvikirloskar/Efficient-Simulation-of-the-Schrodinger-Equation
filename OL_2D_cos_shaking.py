"""
Example simulation of a driven harmonic oscillator in two dimensions (2D) using
the Semi-Global propagation scheme.
"""

# Import standard modules.
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, cast

# Import external modules.
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# Import local modules.
QUEVOLUTIO_ROOT = Path(__file__).resolve().parent / "QuEvolutio"
sys.path.insert(0, str(QUEVOLUTIO_ROOT))
import utils.numerical as numerical
import utils.standard_ho as sho
import utils.visualisation as vis

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RTensor,
    CTensors,
    CSRMatrix,
)
from quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import TDSE, Controls, Hamiltonian
from quevolutio.propagators.semi_global import ApproximationBasis, SemiGlobal

## NOTE: SIMULATION SET UP -----------------------------------------------------


class DHOConstants(QuantumConstants):
    """
    Represents the constants of a 1D driven harmonic oscillator in natural
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
    ???
    unit_cell : float
        The size of the unit cell.
    num_points : int
        The number of spatial discretization points.
    """

    hbar: float = 1.0
    mass: float = 1.0
    v0: float = 5.0
    kl: float = 1.0
    alpha: float = 10.0
    q: float = 0.0
    band: int = 0
    N: int = 10
    unit_cell: float = np.pi / kl
    num_pts: int = 150
    dims: int = 2*N +1 
    L: float = N * unit_cell


class DHOHamiltonian(Hamiltonian):
    """
    Represents the Hamiltonian of a 2D driven harmonic oscillator through the
    "Hamiltonian" interface.

    + Driving Term: f(x, t) = x sin(t)
    + Driving Term: f(y, t) = y sin(t)

    Attributes
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
    eigenvalue_min : float
        The minimum eigenvalue of the Hamiltonian.
    eigenvalue_max : float
        The maximum eigenvalue of the Hamiltonian.
    time_dependent : bool
        A boolean flag that indicates whether the Hamiltonian has explicit time
        dependence.
    ke_time_dependent : bool
        A boolean flag that indicates whether the KE operator has explicit time
        dependence.
    pe_time_dependent : bool
        A boolean flag that indicates whether the PE operator has explicit time
        dependence.

    Internal Attributes
    -------------------
    _ke_operator : RTensor
        The pre-computed kinetic energy operator for the 2D driven harmonic
        oscillator system.
    _pe_operator : RTensor
        The pre-computed standard potential energy operator for the 2D driven
        harmonic oscillator system.
    """

    time_dependent: bool = True
    ke_time_dependent: bool = False
    pe_time_dependent: bool = True

    def __init__(
        self, domain: QuantumHilbertSpace, eigenvalue_min: float, eigenvalue_max: float
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.eigenvalue_min: float = eigenvalue_min
        self.eigenvalue_max: float = eigenvalue_max

        # For static type checking (not runtime required).
        self.domain.constants = cast(DHOConstants, self.domain.constants)

        # Pre-compute the kinetic energy operator.
        self._ke_operator: RTensor = (domain.momentum_meshes[0] ** 2) + (
            domain.momentum_meshes[1] ** 2
        )
        self._ke_operator /= 2.0 * self.domain.constants.mass
        # Pre-compute the potential energy operator.
        self._pe_operator: RTensor = (np.cos(2 * self.domain.constants.kl * self.domain.position_meshes[0])
        ) +  (np.cos(2 * self.domain.constants.kl * self.domain.position_meshes[1])
        )
        self._pe_operator *= -0.5 * self.domain.constants.v0 
    def __call__(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the Hamiltonian on a state. If the Hamiltonian
        has explicit time dependence, a set of controls should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the Hamiltonian has explicit time dependence.

        Returns
        -------
        GVector
            The result of acting the Hamiltonian on the state.
        """

        return self.ke_action(state) + self.pe_action(state, controls)

    def ke_action(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the kinetic energy operator on a state. If the
        kinetic energy operator has explicit time dependence, a set of controls
        should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the kinetic energy operator has explicit time
            dependence.

        Returns
        -------
        GVector
            The result of acting the kinetic energy operator on the state.
        """

        return self.domain.position_space(
            self._ke_operator * self.domain.momentum_space(state)
        )

    def pe_action(self, state: GVector, controls: Optional[Controls] = None) -> GVector:
        """
        Calculates the action of the potential energy operator on a state. If
        the potential energy operator has explicit time dependence, a set of
        controls should be passed.

        Parameters
        ----------
        state : GVector
            The state being acted on. This should have shape
            (*domain.num_points).
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the potential energy operator has explicit time
            dependence.

        Returns
        -------
        GVector
            The result of acting the potential energy operator on the state.
        """
       
        
        
        return (
            self._pe_operator * state + np.cos(controls)
        )


def controls_fn(time: float) -> Controls:
    """
    Evaluates the controls which determine the structure of the time-dependent
    Schrödinger equation (TDSE) for a 2D driven harmonic oscillator system. In
    this case, this is just the time.

    Parameters
    ----------
    time : float
        The time at which to evaluate the controls.

    Returns
    -------
    Controls
        The controls for the TDSE at the given time.
    """

    return time

def H_builder(alpha, N, q, hbar, k_l, m):
  dims = DHOConstants.dims
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

## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: DHOConstants = DHOConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=2,
        num_points=np.array([DHOConstants.num_pts, DHOConstants.num_pts]),
        position_bounds=np.array([[-2 * DHOConstants.unit_cell, 2 * DHOConstants.unit_cell], [-2 * DHOConstants.unit_cell, 2 * DHOConstants.unit_cell]]),
        constants=constants,
    )
    

    hamiltonian = H_builder(DHOConstants.alpha, DHOConstants.N, DHOConstants.q, DHOConstants.hbar, DHOConstants.kl, DHOConstants.mass)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    l = np.arange(-DHOConstants.N, DHOConstants.N + 1)
    c = eigenvectors[:, DHOConstants.band].astype(complex)
    x_grid = np.linspace(-2 * DHOConstants.unit_cell, 2 * DHOConstants.unit_cell, DHOConstants.num_pts)

    phase_basis = np.exp(1j * 2 * DHOConstants.kl * l[:, None] * x_grid[None, :])
    u = np.sum(c[:, None] * phase_basis, axis=0)


    psi = np.exp(1j * DHOConstants.q * x_grid) * u
    psi_real = np.real(psi)
    psi_imag = np.imag(psi)
    psi_imag /= np.max(np.abs(psi_imag))
    psi_real /= np.max(np.abs(psi_real))
    idx = len(x_grid)//2   # x ≈ 0
    if np.real(psi_real[idx]) < 0:
        psi_real *= -1
    psi_real2d = psi_real[:, None] * psi_real[None, :]
    state_initial: RTensor = cast(
        RTensor,
        domain.normalise_state(psi_real2d))
    
    eigenvalue_min = np.min(eigenvalues)
    eigenvalue_max = np.max(eigenvalues)
    # Set up the TDSE.
    hamiltonian: DHOHamiltonian = DHOHamiltonian(domain, eigenvalue_min, eigenvalue_max)
    tdse: TDSE = TDSE(domain, hamiltonian)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=1.0, num_points=1001)

    # Set up the propagator.
    propagator = SemiGlobal(
        tdse,
        time_domain,
        order_m=10,
        order_k=10,
        tolerance=1e-5,
        approximation=ApproximationBasis.NEWTONIAN,
    )

    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")
    # Set a common filename.
    filename: str = "optical_lattice_2d_cos_shaking"

    # Create directories for saving data if they do not exist.
    for folder in (Path("data"), Path("figures"), Path("anims")):
        folder.mkdir(parents=True, exist_ok=True)

    # Save the propagated states.
    np.save(f"data/{filename}.npy", states)

    # Plot the propagated states.
    vis.plot_state_2D(
        states[0],
        domain,
        r"$\text{State} \; (T = 0.00)$",
        f"figures/{filename}_start.png",
    )
    vis.plot_state_2D(
        states[-1],
        domain,
        rf"$\text{{State}} \; (T = {time_domain.time_axis[-1]:.2f})$",
        f"figures/{filename}_final.png",
    )

    # Animate the propagated states.
    vis.animate_states_2D(states, domain, time_domain, f"anims/{filename}.mp4")


if __name__ == "__main__":
    main()

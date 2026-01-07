"""
Example simulation of a driven harmonic oscillator in one dimension (1D) using
the Split-Operator propagation scheme.
"""

# Import standard modules.
import sys
import time
from pathlib import Path
from typing import Optional, cast

# Import external modules.
import numpy as np
from scipy.sparse.linalg import eigsh

# Import local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils.numerical as numerical
import utils.standard_ho as sho
import utils.visualisation as vis

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import Controls, HamiltonianSeparable
from quevolutio.propagators.split_operator import SplitOperator

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
constants = DHOConstants()

class OpticalLatticeHamiltonian(HamiltonianSeparable):
    time_dependent: bool = False
    ke_time_dependent: bool = False
    pe_time_dependent: bool = False

    def __init__(self, domain: QuantumHilbertSpace) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain

        # For static type checking (not runtime required).
        self.domain.constants = cast(DHOConstants, self.domain.constants)

        # Pre-compute the kinetic energy diagonal.
        self._ke_diagonal: RVector = -(domain.momentum_axes[0] ** 2) / (
            2.0 * self.domain.constants.mass
        )
        # Pre-compute the lattice potential energy diagonal.
        self._pe_diagonal: RVector = (
            -0.5
            * self.domain.constants.v0
            * np.cos(2 * self.domain.constants.kl * self.domain.position_axes[0])
        )


    def ke_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
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
       
        return (self._ke_diagonal,)

    def pe_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
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

        return (self._pe_diagonal,)



def controls_fn(time: float) -> Controls:
    """
    Evaluates the controls which determine the structure of the time-dependent
    Schrödinger equation (TDSE) for a 1D driven harmonic oscillator system. In
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
        num_dimensions=1,
        num_points=np.array([DHOConstants.num_pts]),
        position_bounds=np.array([[0, DHOConstants.L]]),
        constants=constants,
    )

    # Set up the initial state.
    h_matrix: CSRMatrix = H_builder(DHOConstants.alpha, DHOConstants.N, DHOConstants.q, DHOConstants.hbar, DHOConstants.kl, DHOConstants.mass)

    hamiltonian = H_builder(DHOConstants.alpha, DHOConstants.N, DHOConstants.q, DHOConstants.hbar, DHOConstants.kl, DHOConstants.mass)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    l = np.arange(-DHOConstants.N, DHOConstants.N + 1)
    c = eigenvectors[:, DHOConstants.band].astype(complex)
    x_grid = np.linspace(0, DHOConstants.L, DHOConstants.num_pts)

    phase_basis = np.exp(1j * 2 * DHOConstants.kl * l[:, None] * x_grid[None, :])
    u = np.sum(c[:, None] * phase_basis, axis=0)

    state_initial = np.exp(1j * DHOConstants.q * x_grid) * u

    state_initial: RVector = cast(
        RVector, domain.normalise_state(state_initial)
    )

    # Set up the Hamiltonian.
    hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)

    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")

    # # Calculate the norms of the states.
    # norms: RVector = numerical.states_norms(states, domain)

    # # Calculate the position expectation values of the states.
    # states_expectation: RVector = cast(
    #     RVector,
    #     np.trapezoid(
    #         ((np.abs(states) ** 2) * domain.position_axes[0]),
    #         dx=domain.position_deltas[0],
    #         axis=1,
    #     ),
    # )

    # # Calculate the error from the exact position expectation values.
    # exact_expectation: RVector = 0.5 * (
    #     (time_domain.time_axis * np.cos(time_domain.time_axis))
    #     - np.sin(time_domain.time_axis)
    # )
    # errors: RVector = np.abs(exact_expectation - states_expectation)

    # Print simulation information.
    # print(f"Runtime \t\t: {(final_time - start_time):.5f} seconds")
    # print(f"Max Error \t\t: {np.max(errors):.5e}")
    # print(f"Max Norm Deviation \t: {np.max(np.abs(norms - norms[0])):.5e}")

    # Set a common filename.
    filename: str = "SplitOperator_optical_lattice"

    # Create directories for saving data if they do not exist.
    for folder in (Path("data"), Path("figures"), Path("anims")):
        folder.mkdir(parents=True, exist_ok=True)

    # Save the propagated states.
    np.save(f"data/{filename}.npy", states)

    # Plot the propagated states.
    vis.plot_state_1D(
        states[0],
        domain,
        r"$\text{State} \; (T = 0.00)$",
        f"figures/{filename}_start.png",
    )
    vis.plot_state_1D(
        states[-1],
        domain,
        rf"$\text{{State}} \; (T = {time_domain.time_axis[-1]:.2f})$",
        f"figures/{filename}_final.png",
    )

    # Animate the propagated states.
    vis.animate_states_1D(states, domain, time_domain, f"anims/{filename}.mp4")


if __name__ == "__main__":
    main()

"""
Example simulation of a driven harmonic oscillator in one dimension (1D) using
the Semi-Global propagation scheme.
"""
# Import standard modules.
import sys
import time
from pathlib import Path
from typing import Optional, cast

# Import external modules.
import numpy as np
import scipy.sparse as sp
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
    omega : float
        The angular frequency of the system.
    """

    hbar: float = 1.0
    mass: float = 1.0
    omega: float = 1.0
    v0: float = 1.0
    kl: float = 1.0

class OpticalLatticeHamiltonian(Hamiltonian):
    time_dependent: bool = False
    ke_time_dependent: bool = False
    pe_time_dependent: bool = False

    def __init__(
        self, domain: QuantumHilbertSpace, eigenvalue_min: float, eigenvalue_max: float
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.eigenvalue_min: float = eigenvalue_min
        self.eigenvalue_max: float = eigenvalue_max

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

            return self.ke_action(state) + self.pe_action(state)

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
            self._ke_diagonal * self.domain.momentum_space(state)
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
            self._pe_diagonal * state
        )

#This function isnt needed as there is no time dependence in the Hamiltonian
# def controls_fn(time: float) -> Controls: 
#     """
#     Evaluates the controls which determine the structure of the time-dependent
#     Schrödinger equation (TDSE) for a 1D driven harmonic oscillator system. In
#     this case, this is just the time.

#     Parameters
#     ----------
#     time : float
#         The time at which to evaluate the controls.

#     Returns
#     -------
#     Controls
#         The controls for the TDSE at the given time.
#     """

#     return time


## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: DHOConstants = DHOConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([512]),
        position_bounds=np.array([[-10.0, 10.0]]),
        constants=constants,
    )

    # Set up the initial state.
    # Here we use the eigenstates of the exact Hamiltonian (T = 0.00).
    h_matrix: CSRMatrix = sho.hamiltonian_matrix(
        domain, constants.mass, [constants.omega]
    )
    eigenvalues, eigenvectors = eigsh(h_matrix, k=5, which="SA")

    state_idx: int = 0
    state_initial: RVector = cast(
        RVector, domain.normalise_state(eigenvectors[:, state_idx])
    )

    # Store the minimum and maximum eigenvalues, accounting for the driving term.
    driving_term: CSRMatrix = cast(
        CSRMatrix, sp.diags(domain.position_axes[0], format="csr")
    )

    # Account for the minimum driving term and restore.
    h_matrix += -driving_term
    eigenvalue_min = eigsh(h_matrix, k=1, which="SA")[0][0]
    h_matrix -= -driving_term

    # Account for the maximum driving term and restore.
    h_matrix += driving_term
    eigenvalue_max = eigsh(h_matrix, k=1, which="LA")[0][0]
    h_matrix -= driving_term

    # Set up the TDSE.
    hamiltonian: DHOHamiltonian = DHOHamiltonian(domain, eigenvalue_min, eigenvalue_max)
    tdse: TDSE = TDSE(domain, hamiltonian)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

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

    # Calculate the norms of the states.
    norms: RVector = numerical.states_norms(states, domain)

    # Calculate the position expectation values of the states.
    states_expectation: RVector = cast(
        RVector,
        np.trapezoid(
            ((np.abs(states) ** 2) * domain.position_axes[0]),
            dx=domain.position_deltas[0],
            axis=1,
        ),
    )

    # Calculate the error from the exact position expectation values.
    exact_expectation: RVector = 0.5 * (
        (time_domain.time_axis * np.cos(time_domain.time_axis))
        - np.sin(time_domain.time_axis)
    )
    errors: RVector = np.abs(exact_expectation - states_expectation)

    # Print simulation information.
    print(f"Runtime \t\t: {(final_time - start_time):.5f} seconds")
    print(f"Max Error \t\t: {np.max(errors):.5e}")
    print(f"Max Norm Deviation \t: {np.max(np.abs(norms - norms[0])):.5e}")

    # Set a common filename.
    filename: str = "driven_ho_1d_testing"

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

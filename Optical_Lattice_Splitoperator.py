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

import QuEvolutio.examples.utils.numerical as numerical
import QuEvolutio.examples.utils.standard_ho as sho
import QuEvolutio.examples.utils.visualisation as vis

# Import QuEvolutio modules.
from QuEvolutio.quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from QuEvolutio.quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from QuEvolutio.quevolutio.core.tdse import Controls, HamiltonianSeparable
from QuEvolutio.quevolutio.propagators.split_operator import SplitOperator
from blochstate1d import GroundBlochState, OLConstants
## NOTE: SIMULATION SET UP -----------------------------------------------------

constants = OLConstants()

class OpticalLatticeHamiltonian(HamiltonianSeparable):
    time_dependent: bool = True
    ke_time_dependent: bool = False
    pe_time_dependent: bool = True

    def __init__(self, domain: QuantumHilbertSpace) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain

        # For static type checking (not runtime required).
        self.domain.constants = cast(OLConstants, self.domain.constants)

        # Pre-compute the kinetic energy diagonal.
        self._ke_diagonal: RVector = (domain.momentum_axes[0] ** 2) / (
            2.0 * self.domain.constants.mass
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
        pe_diag: RVector = -0.5* self.domain.constants.v0 * np.cos((2 * self.domain.constants.kl * self.domain.position_axes[0]) + controls) 

        return (pe_diag,)

   
def make_controls_fn(N_basis, coeffs, omegas, T):
    def controls_fn(time: float) -> Controls:
        crab_function = 0
        for k in range(N_basis // 2):
            crab_function += coeffs[k] * np.cos(omegas[k] * time) #selects the first half of the coefficients for the cosine terms
            crab_function += coeffs[k + N_basis // 2] * np.sin(omegas[k] * time) #selects the second half of the coefficients for the sine terms
        envelope =  (np.sin( (np.pi * time) / T))**2
        phi = (1 + crab_function) * envelope
        return phi
    return controls_fn



## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: OLConstants = OLConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
    psi_real = GroundBlochState().generate_bloch_state()
    state_initial: RVector = cast(
        RVector, domain.normalise_state(psi_real)
    )

    # Set up the Hamiltonian.
    hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)
    #Defining basis
    N_basis = 10
    np.random.seed(0)
    omegas = np.random.uniform(0, 2*np.pi, N_basis)
    coeffs = np.random.uniform(-1, 1, N_basis)
    T = 10.0 #final time
    controls_fn = make_controls_fn(N_basis, coeffs, omegas, T)
    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")
    # Set a common filename.
    filename: str = "SplitOperator_noshaking"

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


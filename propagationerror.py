from Optical_Lattice_Splitoperator import make_controls_fn
from blochstate1d import OLConstants, GroundBlochState
from errors_momentum_space import momentum_space_population, dot_product_error, momentum_space_populationnew
from crab_propagation_tools import OpticalLatticeHamiltonian
import numpy as np
import time
from typing import Optional, cast
from QuEvolutio.quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
from QuEvolutio.quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from crab_propagation_tools import OpticalLatticeHamiltonian, make_controls_fn
from QuEvolutio.quevolutio.propagators.split_operator import SplitOperator
from QuEvolutio.quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    CTensors,
    CSRMatrix,
)
from QuEvolutio.quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from QuEvolutio.quevolutio.core.tdse import TDSE, Controls, Hamiltonian
from QuEvolutio.quevolutio.propagators.semi_global import ApproximationBasis, SemiGlobal
import matplotlib.pyplot as plt
class OpticalLatticeHamiltonianSG(Hamiltonian):
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
        self.domain.constants = cast(OLConstants, self.domain.constants)

        # Pre-compute the kinetic energy diagonal.
        self._ke_diagonal: RVector = (domain.momentum_axes[0] ** 2) / (
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
def controls_fn(time: float):
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

def make_controls_fn(omega):
    def controls_fn(time: float) -> Controls:
        phi = np.sin(omega * time)
        return phi 
    return controls_fn

initial_state = GroundBlochState().generate_bloch_state()
x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]

# des_mom_pop = momentum_space_population(initial_state, x_grid_spacing)
constants: OLConstants = OLConstants()
domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
state_initial: RVector = cast(
        RVector, domain.normalise_state(initial_state)
    )

hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

omegas_list = [0, 1, 2, 5, 10, 20, 50, 100]

all_dt = {}
all_errors = {}

for omega in omegas_list:

    dt_list = []
    errors_split = []
    controls_fn = make_controls_fn(omega)
    time_domain_ref: TimeGrid = TimeGrid(time_min=0.0, time_max=constants.T, num_points=512000)

    propagator = SplitOperator(hamiltonian, time_domain_ref)

    states_ref = propagator.propagate(state_initial, controls_fn, diagnostics=False)
    psi_ref = states_ref[-1]
    ref_mom = momentum_space_population(psi_ref, x_grid_spacing)

    for num_steps in num_steps_list:

        # Time grid
        time_domain = TimeGrid(
            time_min=0.0,
            time_max=constants.T,
            num_points=num_steps
        )

        dt = constants.T / (num_steps - 1)
        dt_list.append(dt)

        # Propagate
        propagator = SplitOperator(hamiltonian, time_domain)

        states = propagator.propagate(
            state_initial, controls_fn, diagnostics=False
        )

        final_state = states[-1]
        final_mom_pop = momentum_space_population(final_state, x_grid_spacing)

        error = dot_product_error(final_mom_pop, ref_mom)
        errors_split.append(error)

    # Store results
    all_dt[omega] = dt_list
    all_errors[omega] = errors_split

for omega in omegas_list:
    data = np.column_stack((all_dt[omega], all_errors[omega]))

    filename = f"error_vs_dt_omega_{omega:.2f}.txt"

    np.savetxt(
        filename,
        data,
        header="dt error",
        comments=""
    )

plt.figure()

for omega in omegas_list:
    plt.loglog(
        all_dt[omega],
        all_errors[omega],
        marker='o',
        label=f'ω = {omega:.2f}'
    )

plt.xlabel('Time step')
plt.ylabel('Percentage error from reference solution')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()
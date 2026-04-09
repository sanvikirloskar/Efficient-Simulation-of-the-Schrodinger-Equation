"""
Example simulation of a standard harmonic oscillator in two dimensions (2D)
using the Split-Operator propagation scheme.
"""

# Import standard modules.
import gc
import sys
import time
from pathlib import Path
from typing import Optional, Sequence, cast

# Import external modules.
import numpy as np
from scipy.sparse.linalg import eigsh

# Import local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import QuEvolutio.examples.utils.numerical as numerical
import QuEvolutio.examples.utils.standard_ho as sho
import QuEvolutio.examples.utils.visualisation as vis

# Import QuEvolutio modules.
from QuEvolutio.quevolutio.core.aliases import (  # isort: skip
    RVector,
    CVector,
    RVectorSeq,
    RTensor,
    CTensors,
    CSRMatrix,
)
from QuEvolutio.quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from QuEvolutio.quevolutio.core.tdse import Controls, HamiltonianSeparable
from QuEvolutio.quevolutio.propagators.split_operator import SplitOperator
from blochstate1d import OLConstants, GroundBlochState
from crab_propagation_tools import make_controls_fn
import matplotlib.pyplot as plt
## NOTE: SIMULATION SET UP -----------------------------------------------------

constants = OLConstants()

class SHOHamiltonian(HamiltonianSeparable):
    """
    Represents the Hamiltonian of a 2D standard harmonic oscillator through the
    HamiltonianSeparable interface.

    Attributes
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the quantum system.
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
    _ke_diagonals : RVector
        The pre-computed kinetic energy diagonals for the 2D standard harmonic
        oscillator system.
    _pe_diagonals : RVector
        The pre-computed potential energy diagonals for the 2D standard
        harmonic oscillator system.
    """

    time_dependent: bool = True
    ke_time_dependent: bool = False
    pe_time_dependent: bool = True

    def __init__(
        self, domain: QuantumHilbertSpace
    ) -> None:
        # Assign attributes.
        self.domain: QuantumHilbertSpace = domain
        self.domain.constants = constants

        # Pre-compute the kinetic energy operator.
        # Pre-compute the kinetic energy diagonals.
        self._ke_diagonals: RVectorSeq = (
            (domain.momentum_axes[0] ** 2) / (2.0 * self.domain.constants.mass),
            (domain.momentum_axes[1] ** 2) / (2.0 * self.domain.constants.mass),
        )

        # # Pre-compute the potential energy operator.
        # self._pe_diagonals: RTensor = (-0.5 * self.domain.constants.v0 * np.cos(2 * self.domain.constants.kl * self.domain.position_axes[0])
        # ),(-0.5 * self.domain.constants.v0 * np.cos(2 * self.domain.constants.kl * self.domain.position_axes[1])
        # )

    def ke_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
        """
        Calculates the kinetic energy diagonal(s) in momentum space. A sequence
        of vectors is returned, where each vector corresponds to each dimension
        in the domain. If the kinetic energy operator has explicit time
        dependence, a set of controls should be passed.

        Parameters
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the kinetic energy operator has explicit time
            dependence.

        Returns
        -------
        RVectorSeq
            The kinetic energy diagonal(s) in momentum space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        
        return (self._ke_diagonals)

    def pe_diagonals(self, controls: Optional[Controls] = None) -> RVectorSeq:
        """
        Calculates the potential energy diagonal(s) in position space. A
        sequence of vectors is returned, where each vector corresponds to each
        dimension in the domain. If the potential energy operator has explicit
        time dependence, a set of controls should be passed.

        Parameters
        ----------
        controls : Optional[Controls]
            The controls which determine the structure of the Hamiltonian. This
            should be passed if the potential energy operator has explicit time
            dependence.

        Returns
        -------
        RVectorSeq
            The potential energy diagonal(s) in position space. This is a
            sequence of GVector, with length (domain.num_dimensions).
        """

        pe_diag: RVectorSeq = (-0.5* self.domain.constants.v0 * np.cos((2 * self.domain.constants.kl * self.domain.position_axes[0]) + controls)), (-0.5* self.domain.constants.v0 * np.cos((2 * self.domain.constants.kl * self.domain.position_axes[0]) + controls))

        return (pe_diag)

def momentum_space_population_2d(psi_xy, dx, dy, num_x_pts, num_y_pts):
    prefactor = dx * dy / (2 * np.pi * constants.hbar)

    psi_p = prefactor * np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(psi_xy))
    )

    psi_p_abs = np.abs(psi_p)**2

    kx = np.fft.fftshift(np.fft.fftfreq(num_x_pts, d=dx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(num_y_pts, d=dy)) * 2*np.pi

    dpx = kx[1] - kx[0]
    dpy = ky[1] - ky[0]

    centre_x = num_x_pts // 2
    centre_y = num_y_pts // 2

    hbark_idx_x = int(round((constants.hbar * constants.kl) / dpx))
    hbark_idx_y = int(round((constants.hbar * constants.kl) / dpy))

    N_basis = constants.N_basis
    mid = N_basis // 2

    mom_pop = np.zeros((constants.n_p, constants.n_p))

    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            idx_x = centre_x + 2*i*hbark_idx_x
            idx_y = centre_y + 2*j*hbark_idx_y
            mom_pop[mid+i, mid+j] = psi_p_abs[idx_x, idx_y]

    total = np.sum(mom_pop)
    if total != 0:
        mom_pop /= total

    return mom_pop

## NOTE: SIMULATION SET UP END -------------------------------------------------


def main():
    # Set up the domain.
    constants: OLConstants = OLConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=2,
        num_points=np.array([constants.num_pts, constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound], [constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
    A_coeffs = np.loadtxt(f"crab_coefficients/best_A_error_0p000994.txt")
    B_coeffs = np.loadtxt(f"crab_coefficients/best_B_error_0p000994.txt")
    omegas =  np.loadtxt(f"crab_coefficients/omegas_0p000994.txt")
    controls_fn = make_controls_fn(constants.N_basis, A_coeffs, B_coeffs, omegas, constants.T, lambda t:0.0)


    # Set up the initial state.
    bloch = GroundBlochState()
    state_initial = bloch.generate_bloch_state(spacial_dims=2)
    eigenvalues, eigenvectors = np.linalg.eigh(bloch.H)
    # state_idx: int = 0
    # state_initial: RTensor = cast(
    #     RTensor,
    #     domain.normalise_state(
    #         eigenvectors[:, state_idx].reshape(
    #             domain.num_points[0], domain.num_points[1]
    #         )
    #     ),
    # )

    # Set up the Hamiltonian.
    hamiltonian: SHOHamiltonian = SHOHamiltonian(domain)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=1001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)

    # Propagate the initial state (timed).
    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn=controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")
    # Set a common filename.
    filename: str = "2D_splitoperator_splitting"

    # Create directories for saving data if they do not exist.
    for folder in (Path("data"), Path("figures"), Path("anims")):
        folder.mkdir(parents=True, exist_ok=True)

    # # Save the propagated states.
    # np.save(f"data/{filename}.npy", states)

    # # Plot the propagated states.
    # vis.plot_state_2D(
    #     states[0],
    #     domain,
    #     r"$\text{State} \; (T = 0.00)$",
    #     f"figures/{filename}_start.png",
    # )
    # vis.plot_state_2D(
    #     states[-1],
    #     domain,
    #     rf"$\text{{State}} \; (T = {time_domain.time_axis[-1]:.2f})$",
    #     f"figures/{filename}_final.png",
    # )

    # # Animate the propagated states.
    # vis.animate_states_2D(states, domain, time_domain, f"anims/{filename}.mp4")
    final_state = states[-1]
    x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
    momentum_populations = momentum_space_population_2d(final_state, x_grid_spacing, x_grid_spacing, constants.num_pts, constants.num_pts)
    orders = np.arange(-10, 12, 2)
    plt.rcParams['font.size'] = 16
    plt.imshow(momentum_populations, origin="lower", extent=[orders[0], orders[-1], orders[0], orders[-1]],
    aspect="equal")
    plt.colorbar(label='Relative momentum population')
    plt.xlabel('Momentum states in x ($\\hbar k$)')
    plt.xticks(orders)
    plt.ylabel('Momentum states in y ($\\hbar k$)')
    plt.yticks(orders)
    plt.tight_layout()
    plt.savefig('2d_split_state.pdf') 
    plt.show()
    print(momentum_populations)

if __name__ == "__main__":
    main()

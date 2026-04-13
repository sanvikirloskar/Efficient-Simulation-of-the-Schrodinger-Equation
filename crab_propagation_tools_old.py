# Import standard modules.
from typing import Optional, cast

# Import external modules.
import numpy as np
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
from blochstate1d_NEW import GroundBlochState, OLConstants
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

def make_controls_fn(N_basis, A_coeffs, B_coeffs, omegas, T):
    def controls_fn(time: float) -> Controls:
        crab_function = 0
        for k in range(N_basis//2): #this should be the same as the number of position points
            crab_function += A_coeffs[k] * np.cos(omegas[k] * time) #selects the first half of the coefficients for the cosine terms
            crab_function += B_coeffs[k] * np.sin(omegas[k] * time) #selects the second half of the coefficients for the sine terms
        envelope =  (np.sin( (np.pi * time) / (T)))**2
        phi = (1 + crab_function) * envelope
        return phi
    return controls_fn


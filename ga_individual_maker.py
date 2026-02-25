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

def make_controls_fn(T,A, B, omega):
    """"
    What you want to do is take the a and b arrays, run a for loop outside conrols fn and call for each element inside the array
    """
    
    def controls_fn(time: float) -> np.ndarray:
        # Compute the shaking values
        ga_vals = A * np.cos(omega * time) + B * np.sin(omega * time)
        
        # Apply envelope
        envelope = (np.sin(np.pi * time / T))**2
        phi = (1 + ga_vals) * envelope
        
        return phi

    return controls_fn
   


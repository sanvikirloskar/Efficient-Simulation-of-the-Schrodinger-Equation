# Import standard modules.
from time import time
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
from blochstate1d import GroundBlochState, OLConstants
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

def make_controls_fn(N_elements, T, N_individuals):
    def controls_fn(time: float) -> Controls:
        #initialising the shaking functions
        shaking_funcs = []
        sigma = 100 #standard deviation for the normal distribution to generate the coefficients from
        A_coeffs = np.random.normal(0, sigma, (N_individuals, N_elements)) #generating the A coefficients from a normal distribution
        B_coeffs = np.random.normal(0, sigma, (N_individuals, N_elements)) #generating the B coefficients from a normal distribution
        omegas = np.random.uniform(0, 0.6, (N_individuals, N_elements)) #generating the frequencies from a uniform distribution between 0 and 0.6
        for i in range(N_individuals):
            ga_vals  = np.zeros(N_elements)
            for j in range(N_elements):
                ga_vals[j] = A_coeffs[i][j] * np.cos(omegas[i][j] * time) + B_coeffs[i][j] * np.sin(omegas[i][j] * time)
            envelope =  (np.sin( (np.pi * time) / T))**2
            phi = (1 + ga_vals) * envelope
            shaking_funcs.append(ga_vals)
        return shaking_funcs, A_coeffs, B_coeffs, omegas
    return controls_fn


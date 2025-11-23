"""
Numerical functions for analysing simulations.
"""

# Import standard modules.
from typing import Optional, cast

# Import external modules.
import numpy as np

# Import QuEvolutio modules.
from quevolutio.core.aliases import RVector, RTensor, GTensors  # isort: skip
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
from quevolutio.core.tdse import Controls, Hamiltonian, TDSEControls


def states_norms(states: GTensors, domain: QuantumHilbertSpace) -> RVector:
    """
    Calculates the norms of a set of states over a given domain. This function
    uses the integral definition (as opposed to vector) of the norm.

    Parameters
    ----------
    states : GTensors
        The states to calculate the norms of. This should have shape
        (time_points, *domain.num_points)
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) that the states are defined on.

    Returns
    -------
    RVector
        The norms of the states.
    """

    # Calculate the norm integral.
    norms: RTensor = np.abs(states) ** 2
    for i in range(domain.num_dimensions):
        norms = cast(
            RTensor,
            np.trapezoid(norms, axis=1, dx=domain.position_deltas[i]),
        )

    return np.sqrt(norms)


def states_energies(
    states: GTensors,
    hamiltonian: Hamiltonian,
    time_domain: TimeGrid,
    controls_fn: Optional[TDSEControls] = None,
) -> RVector:
    """
    Calculates the energy expectation values for a set of states.

    Parameters
    ----------
    states : GTensors
        The states to calculate the energy expectation values of. This should
        have shape (time_points, *domain.num_points)
    hamiltonian : Hamiltonian
        The Hamiltonian of the system.
    time_domain : TimeGrid
        The time domain (grid) over which the states are defined.
    controls_fn : Optional[TDSEControls]
        A callable that generates the controls which determine the
        structure of the TDSE at a given time. This should be passed if the
        Hamiltonian has explicit time dependence.

    Returns
    -------
    RVector
        The energy expectation values of the states.
    """

    # Ensure that a controls callable is passed for a time-dependent system.
    if hamiltonian.time_dependent and controls_fn is None:
        raise ValueError("invalid controls callable")

    # Calculate the energies.
    energies: RVector = np.zeros(time_domain.num_points, dtype=np.float64)
    for i in range(time_domain.num_points):
        # If the Hamiltonian has explicit time dependence, calculate the controls.
        controls: Optional[Controls] = None
        if hamiltonian.time_dependent:
            assert controls_fn is not None
            controls: Optional[Controls] = controls_fn(time_domain.time_axis[i])

        # Calculate the energy.
        energies[i] = np.vdot(states[i], hamiltonian(states[i], controls)).real

    return energies

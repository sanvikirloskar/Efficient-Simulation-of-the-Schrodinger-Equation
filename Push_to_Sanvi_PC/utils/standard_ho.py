"""
Functions for simulating and investigating the standard harmonic oscillator
system.
"""

# Import standard modules.
from typing import Sequence, cast

# Import external modules.
import numpy as np
import scipy.sparse as sp

# Import QuEvolutio modules.
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    CVector,
    GTensor,
    CTensors,
    CSRMatrix,
    LILMatrix,
    CSRMatrixSeq,
)
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid


def hamiltonian_matrix(
    domain: QuantumHilbertSpace, mass: float, omega: Sequence[float]
) -> CSRMatrix:
    """
    Constructs the Hamiltonian for the standard harmonic oscillator in sparse
    matrix format (CSR), for an n-dimensional system. This function uses a
    fourth-order central difference approximation for the kinetic energy
    operator. This function also enforces periodic boundaries.

    Parameters
    ----------
    domain : QuantumHilbertSpace
        The discretised Hilbert space (domain) of the system.
    mass : float
        The mass of the system.
    omega : Sequence[float]
        The angular frequency of the system in each dimension.

    Returns
    -------
    CSRMatrix
        The Hamiltonian for the standard harmonic oscillator in sparse matrix
        format (CSR).
    """

    # Pre-compute constants.
    ke_factor: float = -(domain.constants.hbar**2) / (2.0 * mass)

    # Set the finite-difference coefficients.
    fd_factor_0: float = -30.0
    fd_factor_1: float = 16.0
    fd_factor_2: float = -1.0

    # Construct the Hamiltonian in each dimension.
    hamiltonians: CSRMatrixSeq = []
    for i in range(domain.num_dimensions):
        # Construct the fourth-order central finite-difference stencil.
        fd_diag_0: RVector = np.full(
            domain.num_points[i], fd_factor_0, dtype=np.float64
        )
        fd_diag_1: RVector = np.full(
            (domain.num_points[i] - 1), fd_factor_1, dtype=np.float64
        )
        fd_diag_2: RVector = np.full(
            (domain.num_points[i] - 2), fd_factor_2, dtype=np.float64
        )

        # Construct the kinetic energy operator.
        ke_operator: LILMatrix = cast(  # type: ignore
            LILMatrix,
            sp.diags(
                [fd_diag_0, fd_diag_1, fd_diag_1, fd_diag_2, fd_diag_2],
                offsets=[0, 1, -1, 2, -2],  # type: ignore
                format="lil",
            ),
        )

        # Enforce periodic boundaries.
        ke_operator[0, -1] = fd_factor_1
        ke_operator[-1, 0] = fd_factor_1

        ke_operator[0, -2] = fd_factor_2
        ke_operator[-2, 0] = fd_factor_2
        ke_operator[-1, 1] = fd_factor_2
        ke_operator[1, -1] = fd_factor_2

        # Apply final constants.
        ke_operator *= 1.0 / (12.0 * (domain.position_deltas[i] ** 2.0))
        ke_operator *= ke_factor

        # Convert to CSR format.
        ke_operator: CSRMatrix = ke_operator.tocsr()

        # Construct the potential energy operator.
        pe_diag: RVector = 0.5 * mass * (omega[i] ** 2) * (domain.position_axes[i] ** 2)
        pe_operator: CSRMatrix = cast(CSRMatrix, sp.diags(pe_diag, format="csr"))

        # Store the Hamiltonian.
        hamiltonians.append(ke_operator + pe_operator)

    # Construct the complete Hamiltonian.
    hamiltonian: CSRMatrix = sp.csr_matrix(
        (np.prod(domain.num_points), np.prod(domain.num_points)), dtype=np.float64
    )
    for i in range(domain.num_dimensions):
        # Collect the operators in the current dimension.
        operators: CSRMatrixSeq = []
        for j in range(domain.num_dimensions):
            # If dimensions match, store the Hamiltonian.
            if j == i:
                operators.append(hamiltonians[j])

            # Otherwise, store the identity operator.
            else:
                operators.append(
                    cast(CSRMatrix, sp.eye(domain.num_points[j], format="csr"))
                )

        # Calculate the kronecker product between the operators.
        kronecker_product = operators[0]
        for j in range(1, domain.num_dimensions):
            kronecker_product: CSRMatrix = cast(
                CSRMatrix, sp.kron(kronecker_product, operators[j], format="csr")
            )

        # Update the Hamiltonian.
        hamiltonian += kronecker_product

    return hamiltonian


def eigenstate_solutions(
    state: GTensor, energy: float, hbar: float, time_domain: TimeGrid
) -> CTensors:
    """
    Calculates the analytical time evolution of an eigenstate of the standard
    harmonic oscillator.

    Parameters
    ----------
    state : GTensor
        The eigenstate.
    energy : float
        The energy eigenvalue corresponding to the eigenstate.
    hbar : float
        The reduced Planck's constant.
    time_domain : TimeGrid
        The time domain (grid) over which to calculate the analytical states.

    Returns
    -------
    states : CTensors
        The analytical time evolved states.
    """

    # Calculate the analytical states (vectorised).
    phase_factors: CVector = np.exp(-1j * energy * time_domain.time_axis / hbar)
    states: CTensors = (
        state[np.newaxis, ...] * phase_factors[(...,) + ((np.newaxis,) * state.ndim)]
    )

    return states

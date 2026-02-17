from blochstate1d import GroundBlochState, OLConstants
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


def momentum_space_population(psi_x, x_grid_spacing):
    '''
    Calculates the relative momentum space population for a given real space wavefunction by 
    performing a Fourier transform and finding the absolute square of the momentum space 
    wavefunction. It then finds the population at momentum states separated by 2ℏk.

        Args:
            psi_x: The wavefunction in real space
            x_grid_spacing: The spacing of the position grid in real space

        Returns:
            mom_pop: The relative population of the momentum states from -10ℏk to +10ℏk 
    '''
    prefactor = x_grid_spacing / np.sqrt(2 * np.pi * OLConstants().hbar) # normalisation factor as well as multiplying by dx as fft doesnt do this
    psi_p = prefactor * np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_x)))
    psi_p_abs = np.abs(psi_p) **2
    k = np.fft.fftshift(np.fft.fftfreq(OLConstants().num_pts, d=x_grid_spacing)) * 2*np.pi
    dp = k[1] - k[0]
    mom_pop = np.zeros(OLConstants().n_p) # 11 momentum states from -10hbar*k to + 10hbar*k 
    central_index = OLConstants().num_pts // 2
    HBARK_IND = int(round((OLConstants().hbar * OLConstants().kl) / dp))
    mom_pop[OLConstants().N_basis // 2] = psi_p_abs[central_index]
    for i in range(0, OLConstants.N_basis // 2): 
        # Calculate population at each ±2iℏk using index steps of 2 * HBARK_IND.
        # Populates the central element then moves outwards``
        mom_pop[OLConstants().N_basis // 2 - i] = psi_p_abs[int(central_index - 2 * i * HBARK_IND)]  # negative side
        mom_pop[OLConstants().N_basis // 2 + i] = psi_p_abs[int(central_index + 2 * i * HBARK_IND)]  # positive side
    sum_pop = np.sum(mom_pop)
    if sum_pop != 0:
        mom_pop /= sum_pop # This modifies `mom_pop` in place
    return mom_pop

def dot_product_error(mom_pop, mom_pop_des):
    '''
    Calculates the error of a momentum population from the desired momentum population by finding 
    the dot product between the two momentum populations and dividing by the norms.

    Args:
        mom_pop: The momentum population of the state
        mom_pop_des: The desired momentum population

    Returns:
        error: The error from the desired momentum population

    '''
    dot_product = np.dot(mom_pop, mom_pop_des)
    norm_mom_pop = np.linalg.norm(mom_pop)
    norm_mom_pop_des = np.linalg.norm(mom_pop_des)
    error = (1 - (dot_product / (norm_mom_pop * norm_mom_pop_des))) * 100
    return error


def get_error(coeffs, omegas, N_basis, mom_pop_des = OLConstants().desired_mom_pop):
    '''
    This function gets the error from the desired momentum population for propagating a state with a given 
    shaking function in the CRAB basis.
    
    Args:
        coeffs: The amplitudes of the cosines and sines in the CRAB function
        omegas: The frequencies of the cosines and sines in the CRAB function
        N_basis: The number of basis functions in the CRAB function

    Returns:
        error: The error from the desired momentum population
    '''
    # 1. Set up initial state, propagation and the shaking function

    constants: OLConstants = OLConstants()
    domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
    initial_state = GroundBlochState().generate_bloch_state()
    state_initial: RVector = cast(
        RVector, domain.normalise_state(initial_state)
    )

    # Set up the Hamiltonian.
    hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

    # Set up the time domain.
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)

    # Make the shaking function (controls function), for the given coefficients and frequencies
    controls_fn = make_controls_fn(N_basis, coeffs, omegas, time_domain.time_max)

    # 2. Propagate the initial state (timed).

    print("Propagation Start")
    start_time: float = time.time()
    states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=True
    )
    final_time: float = time.time()
    print("Propagation Done")
    final_state = states[-1]

    # # 3. Find momentum space population

    x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
    mom_pop = momentum_space_population(final_state, x_grid_spacing)

    # 4. Calculate error compared to desired momentum population
    error = dot_product_error(mom_pop, mom_pop_des)

    return error

# 1. Testing the error function with no shaking
omegas1 = np.zeros(OLConstants().N_basis // 2)
coeffs1 = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, 1, 2, 3, 4, 5])

bloch_state_position_space = GroundBlochState().generate_bloch_state()
x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
desired_mom_pop1 = momentum_space_population(bloch_state_position_space, x_grid_spacing)
#Get error from the ground bloch state
error1 = get_error(coeffs1, omegas1, OLConstants().N_basis, mom_pop_des=desired_mom_pop1)

# 2. Testing the error function with shaking
omegas2 = np.random.uniform(0, 2*np.pi, OLConstants().N_basis // 2)
coeffs2 = np.random.uniform(-1, 1, OLConstants().N_basis)
error2 = get_error(coeffs2, omegas2, OLConstants().N_basis)

print(error1)
print(error2)
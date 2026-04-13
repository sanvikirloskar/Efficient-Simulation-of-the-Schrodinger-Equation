import numpy as np
from ga_individual_maker import make_controls_fn1, OpticalLatticeHamiltonian
from blochstate1d_NEW import OLConstants, GroundBlochState
from errors_momentum_space import momentum_space_population
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
from typing import Optional, cast
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
from quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from quevolutio.propagators.split_operator import SplitOperator
import time as time_module
def fitness_error(mom_pop, mom_pop_des):
    """
    Calculates the fitness of an individual by finding the error from the desired momentum population. 
    Used equation 4.3 in Carrie's thesis

    Internal Attributes
    --------------------
    delete_n = indexes in the momentum grid of the desired state
    """
    
    first_term = np.abs(np.sum(mom_pop_des - mom_pop))
    delete1 = [4]
    delete2 = [6]
    
    second_term = np.sum(np.abs(np.delete(mom_pop_des - mom_pop, delete1 + delete2))) 
    third_term = (np.abs(mom_pop[4]-mom_pop[6])/(np.abs(mom_pop[4]) + np.abs(mom_pop[6]) + 1e-10)) # add small number to avoid division by zero

    fitness = first_term + second_term + third_term
    return fitness

def fitness_finder(A_coeffs, B_coeffs, omegas, mom_pop_des):
    """
    Calculates the fitness of an individual by finding the error from the desired momentum population.
    """
    #Initialise the system
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
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=OLConstants.T, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)
    fitnesses = np.zeros(A_coeffs.shape[0])
    # Make the shaking function (controls function), for the given coefficients and frequencies
    for i in range(A_coeffs.shape[0]):
        print(f"Making controls for individual {i+1}")
        A = A_coeffs[i, :] #first half of the coefficients for the cosine terms
        B = B_coeffs[i, :]
        omega = omegas[i, :]
        controls = make_controls_fn1(OLConstants.T, A, B, omega)
    
        
    # 2. Propagate the initial state (timed)
        
        print("Propagation Start")
        start_time: float = time_module.time()
        states: CTensors = propagator.propagate(
            state_initial, controls, diagnostics=False
            )
        final_time: float = time_module.time()
        print("Propagation Done")
        runtime = final_time - start_time
        # print(f"Runtime: {runtime:.2f} seconds")
        final_state = states[-1]

        #find momentum space population
        x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
        mom_pop = momentum_space_population(final_state, x_grid_spacing)

        #calculate fitness
        fitness = fitness_error(mom_pop,mom_pop_des)
        print(f"Fitness for individual {i+1}: {fitness:.6f}")
        fitnesses[i] = fitness
    
    return fitnesses, mom_pop


def sorted_fitnesses(fitnesses, A_coeffs, B_coeffs, omegas):
    """"
    Returns the sorted fitnesses and their corresponding Aand B coefficients. 
    This is used to select the best individuals for the next generation.
    """
    idx = np.argsort(fitnesses)
    sorted_fits = fitnesses[idx]
    sorted_A_coeffs = A_coeffs[idx]
    sorted_B_coeffs = B_coeffs[idx]
    sorted_omegas = omegas[idx]
    return sorted_fits, sorted_A_coeffs, sorted_B_coeffs, sorted_omegas
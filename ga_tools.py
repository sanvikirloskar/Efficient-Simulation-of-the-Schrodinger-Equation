import numpy as np
from ga_individual_maker import make_controls_fn, OpticalLatticeHamiltonian
from blochstate1d import OLConstants, GroundBlochState
from errors_momentum_space import momentum_space_population
from quevolutio.core.domain import QuantumHilbertSpace, TimeGrid
import time
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

def fitness_error(mom_pop, mom_pop_des):
    """
    Calculates the fitness of an individual by finding the error from the desired momentum population. Used equation 4.3 in Carrie's thesis
    """
    first_term = np.abs(mom_pop_des - mom_pop)
    n = 10
    second_term = np.sum(np.abs(np.delete(mom_pop_des - mom_pop, 1))) # need to finish this but i forgot how hannah made the momentum arrays - which element corresponds to which momentum state?

    fitness = first_term + second_term
    return fitness

def fitness_finder(mom_pop_des):
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
    time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)

    # Set up the propagator.
    propagator = SplitOperator(hamiltonian, time_domain)

    # Make the shaking function (controls function), for the given coefficients and frequencies
    controls = make_controls_fn(OLConstants.N_elements, time_domain.time_max, OLConstants.N_individuals)
    fitnesses = np.zeros(OLConstants.N_individuals)
    # 2. Propagate the initial state (timed)
    for i in range(OLConstants.N_individuals):
        controls_fn = controls[i]
        print("Propagation Start")
        start_time: float = time.time()
        states: CTensors = propagator.propagate(
            state_initial, controls_fn, diagnostics=False
            )
        final_time: float = time.time()
        print("Propagation Done")
        runtime = final_time - start_time
        # print(f"Runtime: {runtime:.2f} seconds")
        final_state = states[-1]

        #find momentum space population
        x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
        mom_pop = momentum_space_population(final_state, x_grid_spacing)

        #calculate fitness
        fitness = fitness_error(mom_pop, mom_pop_des)
        print(f"Fitness for individual {i+1}: {fitness:.6f}")
        fitnesses[i] = fitness
    
    return fitnesses


def sorted_fitnesses(fitnesses, A_coeffs, B_coeffs):
    """"
    Returns the sorted fitnesses and their corresponding Aand B coefficients. This is used to select the best individuals for the next generation.
    """
    idx = np.argsort(fitnesses)
    sorted_fitnesses = fitnesses[idx]
    sorted_A_coeffs = A_coeffs[idx]
    sorted_B_coeffs = B_coeffs[idx]
    return sorted_fitnesses, idx, sorted_A_coeffs, sorted_B_coeffs
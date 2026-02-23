import numpy as np
from blochstate1d import OLConstants, GroundBlochState
from crab_propagation_tools import OpticalLatticeHamiltonian
from errors_momentum_space import momentum_space_population
from ga_individual_maker import make_controls_fn
from ga_tools import fitness_finder, sorted_fitnesses

def eugenics(fitnesses, A_coeffs, B_coeffs, omegas, T):
    #Sort the fitnesses and the corresponding coefficients and frequencies
    sorted_fitnesses, sorted_A, sorted_B = sorted_fitnesses(fitnesses, A_coeffs, B_coeffs)

    #Select the best 2 to live (this is what Carrie does)
    best_A = sorted_A[:, :2]
    best_B = sorted_B[:, :2]
    best_omegas = omegas[:2]
    #Select the worst 4 to die (this is what Carrie does)
    worst_A = sorted_A[:, -4:]
    worst_B = sorted_B[:, -4:]
    worst_omegas = omegas[-4:]
    
    #creating a new gen 

def create_new_gen(a):
    a = a
    return a



def mutate(a):
    a = a
    return a



def creep(a):
    a = a
    return a
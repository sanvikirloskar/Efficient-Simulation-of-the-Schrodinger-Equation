import numpy as np
from blochstate1d_NEW import OLConstants, GroundBlochState
from crab_propagation_tools import OpticalLatticeHamiltonian
from errors_momentum_space import momentum_space_population
from ga_individual_maker import make_controls_fn
from ga_tools import fitness_finder, sorted_fitnesses

def eugenics(fitnesses, A_coeffs, B_coeffs, omegas, number_dead):
    """
    Sorts the fitnesses and the corresponding A,B and Omega. Selects the best 2 to live on unaltered
    and sets up the remaining for alteration for the next gen.
    """
    #Sort the fitnesses and the corresponding coefficients and frequencies
    sorted_fits, sorted_A, sorted_B = sorted_fitnesses(fitnesses, A_coeffs, B_coeffs)
    best_fitness = sorted_fits[0]
    #Select the best 2 to live (this is what Carrie does)
    best_A = sorted_A[:2, :]
    best_B = sorted_B[:2, :]
    best_omegas = omegas[:2, :]
    #also separate the middle ones 
    remaining_A = sorted_A[2:-number_dead,:]
    remaining_B = sorted_B[2:-number_dead,:]
    remaining_omegas = omegas[2:-number_dead,:]
    
    return best_A, best_B, best_omegas, remaining_A, remaining_B, remaining_omegas, best_fitness
    


def one_point_crossover(parent1_A, parent2_A, parent1_B, parent2_B, parent1_omegas, parent2_omegas):
    """
    This function takes 2 1D arrays (choose row randomly from the 2D array returned from eugenics) and outputs the 2 child arrays created
    by one point crossover. The crossover point is randomly selected between 1 and N_elements-1 and the 2 arrays are combined to make a 2D array where
    each row is an indivdual with N_elements columns.
    """
    index = np.random.randint(1, parent1_A.shape[0]-1) #randomly select the crossover point
    child_A1 = np.concatenate((parent1_A[:index], parent2_A[index:]))
    child_A2 = np.concatenate((parent2_A[:index], parent1_A[index:]))
    child_B1 = np.concatenate((parent1_B[:index], parent2_B[index:]))
    child_B2 = np.concatenate((parent2_B[:index], parent1_B[index:]))
    child_omegas1 = np.concatenate((parent1_omegas[:index], parent2_omegas[index:]))
    child_omegas2 = np.concatenate((parent2_omegas[:index], parent1_omegas[index:]))
    childA = np.vstack([child_A1, child_A2])
    childB = np.vstack([child_B1, child_B2])
    child_omegas = np.vstack([child_omegas1, child_omegas2])

    return childA, childB, child_omegas

    

def two_point_crossover(parent1_A, parent2_A, parent1_B, parent2_B, parent1_omegas, parent2_omegas):
    """
    Performs 2 point crossover on a given 1D array (select rows from the remaining A,B, Omega) and creates 2 child 1D arrays which 
    are then combined to make a 2D array where each row is an individual with N_elements columns. 
    Crossover points are randomly selected between 1 and N_elements-1
    """
    index1 = np.random.randint(1, parent1_A.shape[0]-2)
    index2 = np.random.randint(index1+1, parent1_A.shape[0]-1)
    child_A1 = np.concatenate((parent1_A[:index1], parent2_A[index1:index2], parent1_A[index2:]))
    child_A2 = np.concatenate((parent2_A[:index1], parent1_A[index1:index2], parent2_A[index2:]))
    child_B1 = np.concatenate((parent1_B[:index1], parent2_B[index1:index2], parent1_B[index2:]))
    child_B2 = np.concatenate((parent2_B[:index1], parent1_B[index1:index2], parent2_B[index2:]))
    child_omegas1 = np.concatenate((parent1_omegas[:index1], parent2_omegas[index1:index2], parent1_omegas[index2:]))
    child_omegas2 = np.concatenate((parent2_omegas[:index1], parent1_omegas[index1:index2], parent2_omegas[index2:]))
    childA = np.vstack([child_A1, child_A2])
    childB = np.vstack([child_B1, child_B2])
    child_omegas = np.vstack([child_omegas1, child_omegas2])

    return childA, childB, child_omegas





def mutate(remaining_A, remaining_B, remaining_omegas):
    """
    Defines a small mutation rate where each gene has a certain probability of being mutated. 
    If an individual is mutated a random gene is selected and changed to a random value between the
    given limit. 
    Designed to be run after eugenics so the best 2 individuals do not get altered.
    Internal Attributes:
    M: int
        The mutation limit. This is the maximum amount by which a gene can be altered during mutation.
    mutation_rate: float
        The probability of mutation for each gene. This is the likelihood that a given gene will be mutated. Between 0 and 1.
    """
    M = 1000  # mutation limit
    mutation_rate = 0.1  # probability of mutation for each gene
    # Create new arrays so we don't change the originals
    child_A = remaining_A.copy()
    child_B = remaining_B.copy()
    child_omegas = remaining_omegas.copy()  # unchanged
    unfit_parents = []  # to keep track of which individuals were mutated
    for i in range(child_A.shape[0]):
        if np.random.rand() < mutation_rate:
            # pick a random gene for this individual
            index = np.random.randint(0, child_A.shape[1])  # columns 0 to N_elements-1
            # mutate A and B
            child_A[i, index] += np.random.uniform(-M, M)
            child_B[i, index] += np.random.uniform(-M, M)
            unfit_parents.append(i)
        else:
            pass  # no mutation, keep the original values
        
    
    return child_A, child_B, child_omegas, unfit_parents


def creep(remaining_A, remaining_B, remaining_omegas, unfit_parents):
    """
    Defines a small creep rate where each gene gets slightly altered. Then this is applied to one gene
    in an individual with a certain probability. Designed to be run after mutation so only
    individuals that were not mutated will be creeped
    
    Interal Attributes:
    C: int
        The creep limit. This is the maximum amount by which a gene can be altered during creep.
    creep_rate: float
        The probability of creep for each gene. This is the likelihood that a given gene will be creeped. Between 0 and 1
    """
    C = 1000  # creep limit
    creep_rate = 0.1  # probability of creep for each gene
    #new arrays so we don't change the original ones 
    child_A = remaining_A.copy()
    child_B = remaining_B.copy()
    child_omegas = remaining_omegas.copy()  # stays unchanged
    unfit_parents = unfit_parents  # to keep track of which individuals were creeped
    for i in range(child_A.shape[0]):
        if i in unfit_parents: 
            continue # only apply creep to individuals that were mutated
            
        if np.random.rand() < creep_rate:
            index = np.random.randint(0, child_A.shape[1])  # pick a random gene
            r = np.random.rand()  # random number between 0 and 1
            delta = (r - 0.5) * 2 * C
            child_A[i, index] += delta
            child_B[i, index] += delta
            unfit_parents.append(i)
        else:
            pass  # no creep, keep the original values
    return child_A, child_B, child_omegas, unfit_parents

def new_gene_maker(remaining_A, remaining_B, remaining_omegas, best_A, best_B, best_omegas, unfit_parents):
    """
    Collates all the new children from crossover, mutation and creep then combines them with the best 2 
    individuals from the previous generation to make the new A, B and Omega arrays for the next generation.
    Designed to be run after mutation and creep, so the unfit parents are not selected for crossover.
    """
    #pick 4 random parents from those who aren't mutated or creeped
    fit_parents = [i for i in range(remaining_A.shape[0]) if i not in unfit_parents]
    selected_parents = np.random.choice(fit_parents, size=4, replace=False)
    parent1_A = remaining_A[selected_parents[0], :]
    parent2_A = remaining_A[selected_parents[1], :]
    parent3_A = remaining_A[selected_parents[2], :]
    parent4_A = remaining_A[selected_parents[3], :]
    parent1_B = remaining_B[selected_parents[0], :]
    parent2_B = remaining_B[selected_parents[1], :]
    parent3_B = remaining_B[selected_parents[2], :]
    parent4_B = remaining_B[selected_parents[3], :]
    parent1_omegas = remaining_omegas[selected_parents[0], :]
    parent2_omegas = remaining_omegas[selected_parents[1], :]
    parent3_omegas = remaining_omegas[selected_parents[2], :]
    parent4_omegas = remaining_omegas[selected_parents[3], :]
    #create 2 children from the 4 parents using crossover
    if np.random.rand() < 0.5:
        childA12, childB12, child_omegas12 = one_point_crossover(parent1_A, parent2_A, parent1_B, parent2_B, parent1_omegas, parent2_omegas)
    else:
        childA12, childB12, child_omegas12 = two_point_crossover(parent1_A, parent2_A, parent1_B, parent2_B, parent1_omegas, parent2_omegas)
    if np.random.rand() < 0.5:
        childA34, childB34, child_omegas34 = one_point_crossover(parent3_A, parent4_A, parent3_B, parent4_B, parent3_omegas, parent4_omegas)
    else:
        childA34, childB34, child_omegas34 = two_point_crossover(parent3_A, parent4_A, parent3_B, parent4_B, parent3_omegas, parent4_omegas)
    # combine the two children
    A_coeffs = np.vstack([childA12, childA34, best_A, remaining_A])
    B_coeffs = np.vstack([childB12, childB34, best_B, remaining_B])
    omegas = np.vstack([child_omegas12, child_omegas34, best_omegas, remaining_omegas])
    return A_coeffs, B_coeffs, omegas
        


def full_genetic_algorithm(A_coeffs, B_coeffs, omegas, mom_pop_des, number_dead):
    """"
    Runs all the above codes to make the full genetic algorithm. It returns the best fitness
    of each generation in a list, as well as the best A,B, Omega from the final generation. 
    Perhaps we set number of iterations super high so that it runs till convergence then the
    best A, B Omega overall will be the best from the final generation.

    Internal Attributes:
    num_iter: int
        The number of generations to run the algorithm for. Set high so that it runs until convergence.
    convergence: float
        The fitness value at which we consider the algorithm to have converged.
          If the best fitness is less than this, we stop the algorithm.

    """
    num_iter = 1000 #remind me to change this if its not over 1000
    print(num_iter)
    print(OLConstants.N_individuals)
    convergence = 0.1 #if the best fitness is less than this, we can stop the algorithm
    best_fit_each_gen = []
    for i in range(num_iter):
        fitnesses, mom_pop = fitness_finder(A_coeffs, B_coeffs, omegas, mom_pop_des)
        best_A, best_B, best_omegas, remaining_A, remaining_B, remaining_omegas, best_fitness = eugenics(fitnesses, A_coeffs, B_coeffs, omegas, number_dead)
        best_fit_each_gen.append(best_fitness)
        print(f"Generation {i}: Best fitness = {best_fitness}")
        if best_fitness < convergence:
            print("Convergence reached.")
            break
        mutated_A, mutated_B, mutated_omegas, unfit_parents = mutate(remaining_A, remaining_B, remaining_omegas)
        creeped_A, creeped_B, creeped_omegas, unfit_parents = creep(mutated_A, mutated_B, mutated_omegas, unfit_parents)
        A_coeffs, B_coeffs, omegas = new_gene_maker(creeped_A, creeped_B, creeped_omegas, best_A, best_B, best_omegas, unfit_parents)
    return best_fit_each_gen, best_A, best_B, best_omegas, mom_pop
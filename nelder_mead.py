import numpy as np
from blochstate1d import GroundBlochState, OLConstants
from crab_propagation_tools import OpticalLatticeHamiltonian, make_controls_fn
from errors_momentum_space import momentum_space_population, get_error
from quevolutio.core.domain import TimeGrid


def sorted_errs(pct, A, B):
    """
    This function aims to get all the errors sorted so that we can easily plug them into Nelder-Mead
    """
    sorted_pct = np.sort(pct)
    #Initialising A and B --> I don't think we should be doing this but I don't know what to replace it with 
    sorted_A = np.zeros_like(A)
    sorted_B = np.zeros_like(B)

    # Populate `sorted_A` and `sorted_B` by matching `pct` with `sorted_pct`
    for i in range(0, len(sorted_pct)):
        # Find the first index where `pct` equals `sorted_pct[i]`     --------> this is Yolan's code
        idx = np.where(pct == sorted_pct[i])[0][0]  
        # Sort A and B according to the current index
        sorted_A[:, i] = A[:, idx]
        sorted_B[:, i] = B[:, idx]
    return sorted_errs, sorted_A, sorted_B

def nelder_mead(sorted_errs, sorted_A, sorted_B, N_basis, omegas, T, A_coeffs, B_coeffs):
    #get constants
    hbar = OLConstants.hbar
    kl = OLConstants.kl

    n_c = OLConstants().N_basis + 1 # number of vertices in our simplex
    n_w = OLConstants().N_basis // 2 # number of omegas, also the number of sines or cosines in our shaking function

    alpha = OLConstants.alpha
    E_r = OLConstants.E_r
    m = OLConstants.mass
    time = TimeGrid(time_min=0.0, time_max=10.0, num_points=10001)
    # we might need other stuff here, but not sure yet

    #creating coeffs for each case - I have just used the vals that Yolan did
    refl_coeff = 1
    exp_coeff = 2
    cont_coeff = 0.5 
    red_coeff = 0.5 

    #here we might need to add a few more params in case we are missing relevant info for the func

    g = make_controls_fn(N_basis, A_coeffs, B_coeffs, omegas, T)

    #initialising empty arrays for each case
    # Reflection
    refl_A = np.zeros((n_c)).T
    refl_B = np.zeros((n_c)).T
    g_refl = np.zeros((len(time))).T  # this needs to be the number of points in the time domain. in yolan's code tsize = len(t) so should be an int
   
    # Expansion
    exp_A = np.zeros((n_c)).T
    exp_B = np.zeros((n_c)).T
    g_exp = np.zeros((len(time))).T
   
    # Contraction
    cont_A = np.zeros((n_c)).T
    cont_B = np.zeros((n_c)).T
    g_cont = np.zeros((len(time))).T
   
    # Reduction
    g_red = np.zeros((len(time), n_w)) #double check that n_w is the correct one here. Yolan uses n_child and n_c differently so ??

    ## BEGIN NELDER-MEAD ##
    bool == 0 #this is a flag for each case. Should set to 0 at the beginning of each iter

    #retrieving relevant errors
    best_err = sorted_errs[0]
    worst_err = sorted_errs[-1]
    next_worst_err = sorted_errs[-2]

    #the coeffs for the worst errors
    worst_A = sorted_A[:, -1]
    worst_B = sorted_B[:, -1]

    #i dont know why we do this but i think its to prevent accidentally overwriting something
    new_A = worst_A.copy()
    new_B = worst_B.copy()
    new_err = worst_err #why don't we copy this?

    #Find the centroid of the matrices
    centroid_A = np.mean(sorted_A[:, : n_c - 1], axis=1)
    centroid_B = np.mean(sorted_B[:, : n_c - 1], axis=1)

    #calculate reflected values of A and B in the centroid : can do this using the formula
    #Formula : x_r = x_0 + refl_coeff(x0 - x_n+1)
    refl_A = centroid_A + refl_coeff*(centroid_A - worst_A)
    refl_B = centroid_B + refl_coeff*(centroid_B - worst_B)

    #propagate with reflected point
    g_refl = make_controls_fn(N_basis, refl_A, refl_B, omegas, T)
    refl_err = get_error(refl_A, refl_B, omegas, N_basis, mom_pop_des = OLConstants().desired_mom_pop)

    #check the conditions for the reflected point
    if refl_err <= worst_err and refl_err > best_err:  #yolan has used next_worst err here while wiki says the worst
        N_calls += 1
        print ('reflection')






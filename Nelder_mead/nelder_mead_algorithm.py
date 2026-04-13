import numpy as np
from blochstate1d import GroundBlochState, OLConstants
from crab_propagation_tools import OpticalLatticeHamiltonian, make_controls_fn
from errors_momentum_space import momentum_space_population, get_error
from QuEvolutio.quevolutio.core.domain import TimeGrid

c = OLConstants()

def sort_errors(errors, A, B):
    """
    Sorts the errors in ascending order and sorts the corresponding A and B coefficients in the same order.

    Args:
        errors: The array of errors to be sorted.
        A: The array of A coefficients corresponding to the errors.
        B: The array of B coefficients corresponding to the errors.

    Returns:
        sorted_errors: The sorted array of errors.
        sorted_A: The array of A coefficients sorted according to the errors.
        sorted_B: The array of B coefficients sorted according to the errors.
    """
    # Retrieve the indices that would sort the errors array
    sorted_indices = np.argsort(errors)
    # Use the sorted indices to sort the errors, A, and B arrays
    sorted_errors = errors[sorted_indices]
    sorted_A = A[:, sorted_indices]
    sorted_B = B[:, sorted_indices]
    return sorted_errors, sorted_A, sorted_B

def nelder_mead(sorted_errs, sorted_A, sorted_B, N_basis, omegas, n_c, n_w, base_pulse):
    """
    Performs one interation of the Nelder-Mead algorithm, following the steps on the Wikipedia page.

    Args: 
        sorted_errs: The errors from the previous iteration, sorted in ascending order.
        sorted_A: The A coefficients from the previous iteration, sorted according to the errors.
        sorted_B: The B coefficients from the previous iteration, sorted according to the errors.
        N_basis: The number of basis functions in the CRAB shaking function.
        omegas: The frequencies of the cosines and sines in the CRAB shaking function.
    Returns:
        sorted_opt_errs: The errors from the current iteration, sorted in ascending order.
        sorted_opt_A: The A coefficients from the current iteration, sorted according to the errors.
        sorted_opt_B: The B coefficients from the current iteration, sorted according to the errors.
        err_NM: The best error from the current iteration.
    """
    # Nelder Mead coefficients
    refl_coeff = 1
    exp_coeff = 2
    cont_coeff = 0.5 
    shrink_coeff = 0.5 

    shrink_flag = 0 
    N_calls = 0
    # Retrieving relevant errors
    best_err = sorted_errs[0]
    worst_err = sorted_errs[-1]
    next_worst_err = sorted_errs[-2]

    # The coeffs for the worst errors
    worst_A = sorted_A[:, -1]
    worst_B = sorted_B[:, -1]

    # The coeffs for the best errors
    best_A = sorted_A[:, 0]
    best_B = sorted_B[:, 0]

    # Find the centroid of the matrices - the mean of A and B excluding the worst point (last column)
    centroid_A = np.mean(sorted_A[:, : n_c - 1], axis=1)
    centroid_B = np.mean(sorted_B[:, : n_c - 1], axis=1)

    # Calculate reflected values of A and B in the centroid : can do this using the formula
    #Formula : x_r = x_0 + refl_coeff(x0 - x_n+1)
    refl_A = centroid_A + refl_coeff * (centroid_A - worst_A)
    refl_B = centroid_B + refl_coeff * (centroid_B - worst_B)

    # Get error for reflected point
    refl_err = get_error(refl_A, refl_B, omegas, N_basis, base_pulse)
    N_calls += 1
    #check the conditions for the reflected point

    #3. Reflection: reflected point better than second worst but not better than best
    if best_err <= refl_err < next_worst_err:
        new_A = refl_A
        new_B = refl_B
        new_err = refl_err

    #4. Expansion: reflected point is the best point so far
    elif refl_err < best_err:
        exp_A = centroid_A + exp_coeff * (refl_A - centroid_A)
        exp_B = centroid_B + exp_coeff * (refl_B - centroid_B)
        exp_err = get_error(exp_A, exp_B, omegas, N_basis, base_pulse)
        N_calls += 1
        #If the expanded point is better than the reflected point, replace the worst point with the expanded point
        if exp_err < refl_err:
            new_A = exp_A
            new_B = exp_B
            new_err = exp_err
        #Otherwise, replace the worst point with the reflected point
        else:
            new_A = refl_A
            new_B = refl_B
            new_err = refl_err

    #5. Contraction: reflected point worse than the second worst point
    else:
        # If the reflected point is better than the worst point, compute the contracted point on the outside
        if refl_err < worst_err:
            cont_A = centroid_A + cont_coeff * (refl_A - centroid_A)
            cont_B = centroid_B + cont_coeff * (refl_B - centroid_B)
            cont_err = get_error(cont_A, cont_B, omegas, N_basis, base_pulse)
            N_calls += 1
            # If the contracted point is better than the reflected point, replace the worst point with the contracted point
            if cont_err < refl_err:
                new_A = cont_A 
                new_B = cont_B
                new_err = cont_err
            # Otherwise, set the flag to go to step 6 (Shrink)
            else:
                shrink_flag = 1
        # If the reflected point is worse than the worst point, compute the contracted point on the inside
        else:
            cont_A = centroid_A + cont_coeff * (worst_A - centroid_A)
            cont_B = centroid_B + cont_coeff * (worst_B - centroid_B)
            cont_err = get_error(cont_A, cont_B, omegas, N_basis, base_pulse)
            N_calls += 1
            # If the contracted point is better than the worst point, replace the worst point with the contracted point
            if cont_err < worst_err:
                new_A = cont_A 
                new_B = cont_B
                new_err = cont_err
            # Otherwise, set the flag to go to step 6 (Shrink)
            else:
                shrink_flag = 1

    #6. Shrink - Replace all points except the best with x_i = x_best + sigma(x_i - x_best)
    if shrink_flag == 1:
        shrink_A = np.zeros((n_w, n_c))
        shrink_A[:, 0] = best_A
        shrink_B = np.zeros((n_w, n_c))
        shrink_B[:, 0] = best_B
        for i in range(1, n_c):
            shrink_A[:, i] = best_A + shrink_coeff * (sorted_A[:, i] - best_A)
            shrink_B[:, i] = best_B + shrink_coeff * (sorted_B[:, i] - best_B)
        optimised_A = shrink_A
        optimised_B = shrink_B
        optimised_errs = np.zeros(n_c)

        # keep the best error unchanged
        optimised_errs[0] = best_err

        for i in range(1, n_c):
            optimised_errs[i] = get_error(
            optimised_A[:, i], optimised_B[:, i], omegas, N_basis, base_pulse)
            N_calls += 1
    
    else:
        # Replace the worst point with the new point (reflected, expanded, or contracted)
        optimised_A = np.copy(sorted_A)
        optimised_B = np.copy(sorted_B)
        optimised_errs = np.copy(sorted_errs)
        optimised_A[:, -1] = new_A
        optimised_B[:, -1] = new_B
        optimised_errs[-1] = new_err

    sorted_opt_errs, sorted_opt_A, sorted_opt_B = sort_errors(optimised_errs, optimised_A, optimised_B)

    # Extract the best error (first element in sorted_errs)
    err_NM = sorted_opt_errs[0]

    return sorted_opt_errs, sorted_opt_A, sorted_opt_B, err_NM, N_calls









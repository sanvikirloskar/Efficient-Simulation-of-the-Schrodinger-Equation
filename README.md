Repository containing python code for the MSci Project
The main branch contains important pieces of code e.g working lattices in each dimension.
Our respective branches have rough code that might not solve previous problems but acts as a record of the work which was done. Some of this code does not work.
GitHub repository link: https://github.com/sanvikirloskar/Efficient-Simulation-of-the-Schrodinger-Equation

Contents of Main branch Code:
--------------------------
1. Cosine Potential.ipynb : Solved the TISE for the optical lattice. Produces the energy bands of the lattice
2. 1D_SEMIGLOBAL_MAIN.ipynb: Clean simulation of the 1D TDSE of optical lattice. Uploaded version has no shaking but functions can be imported from crab_propagation_tools, crab_propagation_tools_semiglobal, or ga_individual_maker
3. 2D_SEMIGLOBAL_MAIN.ipynb: Clean simulation of the 2D TDSE of the optical lattice. Uploaded version has no shaking but functions can be imported from crab_propagation_tools, crab_propagation_tools_semiglobal, or ga_individual_maker
4. OL_visualisation.py: Adapted version of utils.visualisation in QuEvolutio. It produces the same figs with bigger fonts
5. blochstate1d_NEW.py: Similar to blochstate1d, this contains the OOP list of the main constants as well as the function used to generate Bloch states before propagating. Can be done in 1 or 2 dimensions. The difference between blochstate1d and blochstate1D_NEW is that _NEW contains elements required for semi-global that are not used in split-operator.
6. crab_propagation_tools_old.py: An older version of crab_propagation_tools. As we both used this differently, this was edited to account for dCRAB by Hannah. However, Sanvi still used functions in the old one so this is labeled as _old and uploaded
7. crab_propagation_tools_semiglobal: Contains the versions of propagation tools required to run semi-global scheme. Requires blochstate1d_NEW
8. ga_2D_momstates.py: Contains code used to find the momentum states after 2D propagation. 
9. ga_analysis.ipynb: Used to generate graphs for the shaking function, the 1D final state and splitting over time
10. ga_full_sim.ipynb: The code where the full genetic algorithm was run, including test funcs, genetic algorithm and progression curve
11. ga_graphs: The code where fitnesses from ga_full_sim were collated and put into graphs for the report
12. ga_individual_maker.py: Contains the code to make the individuals when given the basis elements - Fourier amplitudes + frequencies. A parametrised version of the code is in the comments of def make_controls_fn1, but is left as a comment as data was not produced with it. Also contains kinetic and potential terms for propagation methods as required by QuEvolutio. 
13. ga_tools.py: Contains the fitness function, the fitness finder after propagating, and the sorting algorithm for each fitness
14. genetic_algorithm.py: Contains the elitism functions, mutation,creep and crossover functions, and collates these + ga_tools to create a full_genetic_algorithm function
15. NC_error_graphs: The graphs made for the report comparing errors in Newtonian/Chebyshev expansions, errors vs propagation time and errors vs number of Brillouin zones


Data Included:
--------------
Semiglobal data:
GA data: best fitnesses for all $r_m$ variations recorded, fitnesses for all $A_die$ variations recorded, example of a shaking function with fitness 0.11 (best_solution.npz)
Nelder-Mead data:

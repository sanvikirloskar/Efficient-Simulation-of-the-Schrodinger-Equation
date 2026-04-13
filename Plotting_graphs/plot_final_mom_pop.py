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
from crab_propagation_tools import OpticalLatticeHamiltonian, make_controls_fn, make_total_base_pulse
from QuEvolutio.quevolutio.propagators.split_operator import SplitOperator
from errors_momentum_space import momentum_space_population, get_error
import matplotlib.pyplot as plt
constants: OLConstants = OLConstants()

plt.rcParams['font.size'] = 16
optimised_A = np.loadtxt(f"crab_coefficients/best_A_error_0p000994.txt")
optimised_B = np.loadtxt(f"crab_coefficients/best_B_error_0p000994.txt")
omegas =  np.loadtxt(f"crab_coefficients/omegas_0p000994.txt")

base_pulses = []
base_pulse = make_total_base_pulse(base_pulses)

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

hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=constants.T, num_points=10001)

propagator = SplitOperator(hamiltonian, time_domain)

controls_fn = make_controls_fn(constants.N_basis, optimised_A, optimised_B, omegas, constants.T, base_pulse)

states: CTensors = propagator.propagate(
    state_initial, controls_fn, diagnostics=False
    )
final_state = states[-1]

x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
mom_pop = momentum_space_population(final_state, x_grid_spacing)
momentum_idx = np.arange(-10, 12, 2)

print(mom_pop)
other_mom_pop = 1 - mom_pop[4] - mom_pop[6]
print(other_mom_pop)

plt.figure()
plt.bar(momentum_idx, mom_pop)
plt.xlabel('Momentum state ($\\hbar k$)')
plt.xticks(momentum_idx)
plt.ylabel('Relative momentum population')
plt.grid(True, alpha=0.3)

target_x = [-2, 2]
target_height = [0.5, 0.5]

plt.bar(
    target_x,
    target_height,
    width=0.8,              
    edgecolor='black',
    facecolor='none',     
    linewidth=2,        
    label='Target'
)

plt.tight_layout()
plt.savefig('relative_momentum_population.pdf') 
plt.show()
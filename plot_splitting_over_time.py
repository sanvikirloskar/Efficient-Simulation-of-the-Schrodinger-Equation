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
from errors_momentum_space import momentum_space_population
import matplotlib.pyplot as plt
constants: OLConstants = OLConstants()

initial_state = GroundBlochState().generate_bloch_state()
x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]

constants: OLConstants = OLConstants()
domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=1,
        num_points=np.array([constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
state_initial: RVector = cast(
        RVector, domain.normalise_state(initial_state)
    )

hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

A_coeffs = np.loadtxt(f"crab_coefficients/best_A_error_0p000994.txt")
B_coeffs = np.loadtxt(f"crab_coefficients/best_B_error_0p000994.txt")
omegas =  np.loadtxt(f"crab_coefficients/omegas_0p000994.txt")
base_pulses = []
base_pulse = make_total_base_pulse(base_pulses)
controls_fn = make_controls_fn(constants.N_basis, A_coeffs, B_coeffs, omegas, constants.T, base_pulse)
time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=constants.T, num_points=10001)
propagator = SplitOperator(hamiltonian, time_domain)

states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=False
    )

n_samples = 2000
sample_indices = np.linspace(0, len(states) - 1, n_samples, dtype=int)
n_p = constants.n_p
populations = np.zeros((n_p, n_samples))

for j, i in enumerate(sample_indices):
    psi = states[i]
    populations[:, j] = momentum_space_population(psi, x_grid_spacing)
time_grid = np.linspace(0.0, constants.T, len(states))
time_sampled= time_grid[sample_indices]

# Padding the population array to create gaps between momentum states in the plot
factor = 2  

n_p, n_t = populations.shape
pop_padded = np.zeros((n_p * factor, n_t))


momentum_vals = np.arange(-10, 11, 2)  # [-10, -8, ..., 10]
n_p = len(momentum_vals)

pop_padded = np.zeros((n_p * factor, n_t))
for i in range(n_p):
    pop_padded[i * factor] = populations[i]

extent = [
    time_sampled[0],
    time_sampled[-1],
    momentum_vals[0] - 0.5,
    momentum_vals[-1] + 0.5,
]


plt.rcParams['font.size'] = 16
plt.imshow(
    pop_padded,
    aspect='auto',
    origin='lower',
    extent=extent,
    cmap='viridis'
)

plt.yticks(momentum_vals)
plt.ylim(momentum_vals[0] - 0.5, momentum_vals[-1] + 0.5)
plt.xlabel('Time (simulation units)')
plt.ylabel(r'Momentum state ($\hbar k$)')
plt.colorbar(label='Relative momentum population')
plt.tight_layout()
plt.savefig('splitting_over_time.pdf') 
plt.show()
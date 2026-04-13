from blochstate1d import OLConstants, GroundBlochState
import matplotlib.pyplot as plt
from errors_momentum_space import momentum_space_population
import numpy as np
plt.rcParams['font.size'] = 15
ground_bloch_state = GroundBlochState().generate_bloch_state(band=0)
excited_bloch_state = GroundBlochState().generate_bloch_state(band=1)
x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
mom_pop_0 = momentum_space_population(ground_bloch_state, x_grid_spacing)
mom_pop_1 = momentum_space_population(excited_bloch_state, x_grid_spacing)
momentum_idx = np.arange(-10, 12, 2)
width = 0.8

plt.figure()
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_axisbelow(True)

plt.bar(momentum_idx - width/2, mom_pop_0, width=0.8)
plt.bar(momentum_idx + width/2, mom_pop_1, width=0.8)

plt.xlabel('Momentum states ($\\hbar k_L$)')
plt.ylabel('Relative momentum population')
plt.xticks(momentum_idx)


plt.tight_layout()
plt.savefig('quantised_momentum.pdf') 

plt.show()

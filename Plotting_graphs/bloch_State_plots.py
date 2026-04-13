from blochstate1d import OLConstants, GroundBlochState
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
constants = OLConstants()
state_hk = GroundBlochState().generate_bloch_state(q=1)
state = GroundBlochState().generate_bloch_state(q=0)
x_grid = np.linspace(constants.lower_x_bound, constants.upper_x_bound, constants.num_pts)
plt.plot(x_grid, state)
plt.plot(x_grid, state_hk)
plt.xlabel('x', fontsize=16)
plt.ylabel(r'$\psi(x)$', fontsize=16)
plt.legend(['q=0', r'$q=\hbar k_l$'], fontsize=12)
plt.tight_layout()
plt.savefig('bloch_states_alpha20.pdf')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from crab_propagation_tools_old import make_controls_fn
from blochstate1d_NEW import OLConstants, GroundBlochState
from ga_individual_maker import make_controls_fn1
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
from crab_propagation_tools_old import OpticalLatticeHamiltonian
from quevolutio.propagators.split_operator import SplitOperator

import matplotlib.pyplot as plt


def momentum_space_population_2d(psi_xy, dx, dy, num_x_pts, num_y_pts):
    prefactor = dx * dy / (2 * np.pi * constants.hbar)

    psi_p = prefactor * np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(psi_xy))
    )

    psi_p_abs = np.abs(psi_p)**2

    kx = np.fft.fftshift(np.fft.fftfreq(num_x_pts, d=dx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(num_y_pts, d=dy)) * 2*np.pi

    dpx = kx[1] - kx[0]
    dpy = ky[1] - ky[0]

    centre_x = num_x_pts // 2
    centre_y = num_y_pts // 2

    hbark_idx_x = int(round((constants.hbar * constants.kl) / dpx))
    hbark_idx_y = int(round((constants.hbar * constants.kl) / dpy))

    N_basis = constants.N_basis
    mid = N_basis // 2

    mom_pop = np.zeros((constants.n_p, constants.n_p))

    for i in range(-mid, mid+1):
        for j in range(-mid, mid+1):
            idx_x = centre_x + 2*i*hbark_idx_x
            idx_y = centre_y + 2*j*hbark_idx_y
            mom_pop[mid+i, mid+j] = psi_p_abs[idx_x, idx_y]

    total = np.sum(mom_pop)
    if total != 0:
        mom_pop /= total

    return mom_pop


#load relevant data
constants: OLConstants = OLConstants()
data = np.load('best_solution.npz')
A = data['A'][0,:]
B = data['B'][0,:]
omegas = data['omegas'][0,:]

#Set up the initial state
constants: OLConstants = OLConstants()
domain: QuantumHilbertSpace = QuantumHilbertSpace(
        num_dimensions=2,
        num_points=np.array([constants.num_pts, constants.num_pts]),
        position_bounds=np.array([[constants.lower_x_bound, constants.upper_x_bound], [constants.lower_x_bound, constants.upper_x_bound]]),
        constants=constants,
    )
state_idx: int = 0
initial_state = GroundBlochState().generate_bloch_state()
state_initial: RVector = cast(
        RVector, domain.normalise_state(initial_state)
    )

hamiltonian: OpticalLatticeHamiltonian = OpticalLatticeHamiltonian(domain)

#controls_fn = make_controls_fn(OLConstants.N_elements,A, B, omegas,OLConstants.T)
controls_fn = make_controls_fn1(OLConstants.T, A, B, omegas)
time_domain: TimeGrid = TimeGrid(time_min=0.0, time_max=constants.T, num_points=10001)
propagator = SplitOperator(hamiltonian, time_domain)

states: CTensors = propagator.propagate(
        state_initial, controls_fn, diagnostics=False
    )

final_state = states[-1]
x_grid_spacing = GroundBlochState().x_grid[1] - GroundBlochState().x_grid[0]
momentum_populations = momentum_space_population_2d(final_state, x_grid_spacing, x_grid_spacing, constants.num_pts, constants.num_pts)
orders = np.arange(-10, 12, 2)
plt.rcParams['font.size'] = 16
plt.imshow(momentum_populations, origin="lower", extent=[orders[0], orders[-1], orders[0], orders[-1]],
aspect="equal")
plt.colorbar(label='Relative momentum population')
plt.xlabel('Momentum states in x ($\\hbar k$)')
plt.xticks(orders)
plt.ylabel('Momentum states in y ($\\hbar k$)')
plt.yticks(orders)
plt.tight_layout()
plt.savefig('2d_split_state_ga.pdf') 
plt.show()
print(momentum_populations)
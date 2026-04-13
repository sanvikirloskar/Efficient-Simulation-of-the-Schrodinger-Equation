from blochstate1d import GroundBlochState
from blochstate1d import OLConstants
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
colors = ['blue', 'orange']
cmap_custom = LinearSegmentedColormap.from_list('blue_orange', colors)
plt.rcParams['font.size'] = 16
x_grid = GroundBlochState().x_grid
y_grid = np.linspace(-OLConstants().unit_cell, OLConstants().unit_cell, OLConstants().num_pts)
#Plot the ground state Bloch wavefunction in 1d
psi = GroundBlochState().generate_bloch_state()
plt.plot(x_grid, np.real(psi))
plt.xlabel('x')
plt.ylabel(r'$\psi(x)$')
plt.show()

#Plot the ground state Bloch wavefunction in 2d with contours of lattice potential
cos_x = np.cos(2 * OLConstants().kl * x_grid)
cos_y = np.cos(2 * OLConstants().kl * y_grid)
lattice_potential = -cos_x[:, None] - cos_y[None, :]
psi_real2d = np.real(GroundBlochState().generate_bloch_state(spacial_dims=2))
plt.figure(figsize=(7,6))

# Heatmap of density
plt.imshow(
    psi_real2d,
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    origin='lower',
    cmap='RdYlBu_r'
    # cmap='coolwarm'
)

plt.colorbar(label=r"Amplitude of $\psi(x,y)$")

# Contours of potential
plt.contour(
    x_grid, y_grid, lattice_potential,
    levels=15,
    colors='white',
    linewidths=0.8
)

plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig('2dlattice.pdf') 
plt.show()

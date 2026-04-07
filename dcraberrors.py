import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
files = glob.glob("crab_coefficients/best_errors_10dcrab*.txt") # with dcrab
# files = glob.glob("crab_coefficients/best_errors_*.txt") #without dcrab
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


fig, ax = plt.subplots()

# Main plot
for file in files:
    errors = np.loadtxt(file)
    iterations = np.arange(len(errors))
    ax.plot(iterations, errors, linewidth=1.2)

ax.set_xlabel("Iteration")
ax.set_ylabel("Error (%)")
ax.grid(True, alpha=0.3)

ax_inset = inset_axes(ax, width="40%", height="40%", loc="upper right")

for file in files:
    errors = np.loadtxt(file)
    iterations = np.arange(len(errors))

    mask = (iterations >= 50) & (iterations <= 200)
    ax_inset.plot(iterations[mask], errors[mask], linewidth=1.2)

ax_inset.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('optimisationdcrab.pdf') 
plt.show()
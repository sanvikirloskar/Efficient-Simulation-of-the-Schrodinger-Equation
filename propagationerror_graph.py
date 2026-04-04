import numpy as np
import matplotlib.pyplot as plt

omegas_list = [0, 1, 2, 5, 10, 20, 50, 100] # MUST match what you used before

plt.figure()

for omega in omegas_list:
    filename = f"error_vs_dt_omega_{omega:.2f}.txt"
    
    data = np.loadtxt(filename, skiprows=1)  # Skip the header row
    
    dt = data[:, 0]
    error = data[:, 1]

    plt.loglog(
        dt,
        error,
        marker='o',
        label=f'ω = {omega:.2f}'
    )

plt.xlabel('Time step')
plt.ylabel('Percentage error from reference solution')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()
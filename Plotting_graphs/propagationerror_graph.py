import numpy as np
import matplotlib.pyplot as plt

omegas_list = [0, 1, 2, 5, 10, 20, 50, 100] 

plt.figure()

for omega in omegas_list:
    filename = f"error_vs_dt_omega_{omega:.2f}.txt"
    
    data = np.loadtxt(filename, skiprows=1)  
    
    dt = data[:, 0]
    error = data[:, 1]

    plt.loglog(
        dt,
        error,
        marker='o',
        markersize=4,
        linewidth=1,
        label=f'ω = {omega:.2f}'
    )
plt.xlim(1e-5, None)
plt.xlabel('Time step', fontsize=13)
plt.ylabel('Percentage error from reference solution', fontsize=13)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('propagationerror.pdf') 
plt.show()
# Import standard modules.
from typing import Optional, cast

# Import external modules.
import numpy as np
# Import QuEvolutio modules.
from QuEvolutio.quevolutio.core.aliases import (  # isort: skip
    RVector,
    GVector,
    RVectorSeq,
    CTensors,
    CSRMatrix,
)
from QuEvolutio.quevolutio.core.domain import QuantumConstants, QuantumHilbertSpace, TimeGrid
from QuEvolutio.quevolutio.core.tdse import Controls, HamiltonianSeparable
from blochstate1d import GroundBlochState, OLConstants
import matplotlib.pyplot as plt
A_coeffs = np.loadtxt(f"crab_coefficients/best_A_error_0p000994.txt")
B_coeffs = np.loadtxt(f"crab_coefficients/best_B_error_0p000994.txt")
omegas =  np.loadtxt(f"crab_coefficients/omegas_0p000994.txt")

def make_controls_fn(N_basis, A_coeffs, B_coeffs, omegas, T):
    def controls_fn(time: float) -> Controls:
        crab_function = 0
        for k in range(N_basis // 2):
            crab_function += A_coeffs[k] * np.cos(omegas[k] * time) #selects the first half of the coefficients for the cosine terms
            crab_function += B_coeffs[k] * np.sin(omegas[k] * time) #selects the second half of the coefficients for the sine terms
        envelope =  (np.sin( (np.pi * time) / T))**2
        phi = (1 + crab_function) * envelope
        return phi 
    return controls_fn
N_basis = 10
shaking_func = make_controls_fn(N_basis, A_coeffs, B_coeffs, omegas, T=10.0)
time_domain = np.linspace(0.0, 10.0, 10001)

phi_values = np.array([shaking_func(t) for t in time_domain])
# real_time = np.linspace(0.0, 250, 10001)


plt.rcParams['font.size'] = 16
plt.figure()
plt.plot(time_domain, phi_values,linewidth=1.5, color='tab:blue')
plt.xlabel("Time (simulation units)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('shaking_function.pdf') 
plt.show()



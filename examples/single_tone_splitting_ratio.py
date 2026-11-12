"""
examples/single_tone_splitting_ratio.py
=======================================
Single ring (N_r = 1), single RF tone (N_f = 1), two modes (N = 2):
sweep |kappa_1| from 0 to KAPPA_MAX (units of gamma_e) and track the 2x2
splitting ratio.

Closed form (Δω = 0, γ_i = 0):
    M_00 = (1 - κ²)/(1 + κ²),   M_10 = -2iκ/(1 + κ²)
so  T_cross = 4κ²/(1 + κ²)²  ->  50:50 at κ = √2 - 1,  full swap at κ = 1.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import cascaded_scattering_matrix, unitarity_error

# ── CONFIG ───────────────────────────────────────────────────────────────────
KAPPA_MAX = 1.0   # max RF coupling strength (units of gamma_e)
N_PTS = 401
GAMMA_I = 0.0     # intrinsic loss (units of gamma_e)
PHASE = 0.0       # RF tone phase -- only changes the cross-term phase

kappas = np.linspace(0, KAPPA_MAX, N_PTS)

# ── Sweep ────────────────────────────────────────────────────────────────────
M = np.array([cascaded_scattering_matrix([[k * np.exp(1j * PHASE)]], N=2, gamma_i=GAMMA_I)
              for k in kappas])
T_bar = np.abs(M[:, 0, 0]) ** 2
T_cross = np.abs(M[:, 1, 0]) ** 2

# Self-check against the closed form (lossless only)
if GAMMA_I == 0:
    assert np.allclose(T_cross, 4 * kappas**2 / (1 + kappas**2) ** 2)
    assert max(unitarity_error(m) for m in M) < 1e-12

k50 = kappas[np.argmin(np.abs(T_cross - 0.5))]
print(f"50:50 at kappa ≈ {k50:.4f}  (analytic √2-1 = {np.sqrt(2) - 1:.4f})")
print(f"Cross at kappa_max = {KAPPA_MAX}: {T_cross[-1]:.4f}")

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.plot(kappas, T_bar, label=r"bar $|M_{00}|^2$")
plt.plot(kappas, T_cross, label=r"cross $|M_{10}|^2$")
plt.plot(kappas, 1 - T_bar - T_cross, "k:", label="loss")
plt.axvline(np.sqrt(2) - 1, color="grey", ls="--", lw=0.8)
plt.xlabel(r"$|\kappa_1|/\gamma_e$")
plt.ylabel("Power")
plt.title(r"Single ring, single tone, 2$\times$2")
plt.legend()
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/single_tone_splitting_ratio.png", dpi=120)
plt.show()

"""
demo.py — freq_beamsplitter package example
Reproduces the demo cases from Buddhiraju et al. (2021).
"""

import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import (
    scattering_matrix,
    cascaded_scattering_matrix,
    unitarity_error,
    fidelity,
    plot_matrix,
    plot_matrix_grid,
)

N_sb = 2   # modes {-2,-1,0,+1,+2}  →  5×5 matrix
N_f  = 4   # number of modulation tones

rng   = np.random.default_rng(42)
kappa = rng.random(N_f) * 0.5 + 1j * rng.random(N_f) * 0.5

# ── Case 1: lossless, on-resonance (should be unitary) ──────────────────────
M_ideal = scattering_matrix(kappa, N_sb)
print(f"Case 1 — lossless, on-resonance:")
print(f"  ||M†M - I||_F = {unitarity_error(M_ideal):.2e}  (expect ~1e-15)\n")

# ── Case 2: intrinsic loss γ_i = 0.05 γ_e ───────────────────────────────────
M_lossy = scattering_matrix(kappa, N_sb, gamma_i=0.05)
print(f"Case 2 — γ_i = 0.05 γ_e:")
print(f"  Unitarity error = {unitarity_error(M_lossy):.4f}  (expect > 0)")
print(f"  Max |M_ij|      = {np.max(np.abs(M_lossy)):.4f}\n")

# ── Case 3: non-zero detuning Δω = 0.5 γ_e ──────────────────────────────────
M_det = scattering_matrix(kappa, N_sb, delta_omega=0.5)
print(f"Case 3 — detuning Δω = 0.5 γ_e:")
print(f"  Unitarity error = {unitarity_error(M_det):.2e}\n")

# ── Case 4: 4 cascaded rings ─────────────────────────────────────────────────
N_r = 4
kappa_list = [rng.random(N_f) * 0.5 + 1j * rng.random(N_f) * 0.5 for _ in range(N_r)]
M_casc = cascaded_scattering_matrix(kappa_list, N_sb)
print(f"Case 4 — {N_r} cascaded rings, lossless:")
print(f"  Unitarity error = {unitarity_error(M_casc):.2e}\n")

# ── Fidelity to DFT target ───────────────────────────────────────────────────
N   = 2 * N_sb + 1
idx = np.arange(N)
U_dft = np.exp(2j * np.pi * np.outer(idx, idx) / N) / np.sqrt(N)
print(f"Fidelity of 4-ring cascade to DFT: {fidelity(U_dft, M_casc):.4f}")
print("  (random κ_l — optimise via inverse design to improve)\n")

# ── Visualisation ────────────────────────────────────────────────────────────
fig_grid = plot_matrix_grid(
    [M_ideal, M_lossy, M_det, M_casc],
    ["Lossless", "γ_i = 0.05 γ_e", "Δω = 0.5 γ_e", "4 rings (cascaded)"],
)
plt.tight_layout()
plt.show()

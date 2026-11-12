"""
examples/single_tone_splitting_ratio.py
=======================================
Can a single ring (N_r = 1) driven by a single RF tone (N_f = 1) act as a
tunable frequency beamsplitter?  Sweep |kappa_1| from 0 to KAPPA_MAX (units
of gamma_e) and track the splitting ratio between two adjacent modes.

Two-mode case (N = 2, Δω = 0, γ_i = 0) has a closed form:
    M_00 = (1 - κ²)/(1 + κ²),   M_01 = -2iκ/(1 + κ²)
so  T_cross = 4κ²/(1 + κ²)²  ->  50:50 at κ = √2 - 1,  full swap at κ = 1.

N = 2 assumes the comb is truncated to two modes (e.g. by an aux ring /
dispersion). An untruncated comb (N_WIDE) leaks power into outer sidebands,
shown for comparison with the input in the centre mode.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from freq_beamsplitter import scattering_matrix, unitarity_error

# ── CONFIG ───────────────────────────────────────────────────────────────────
KAPPA_MAX = 1.0   # max RF coupling strength (units of gamma_e)
N_PTS = 401
GAMMA_I = 0.0     # intrinsic loss (units of gamma_e)
PHASE = 0.0       # RF tone phase -- only changes the cross-term phase
N_WIDE = 9        # untruncated-comb comparison (odd so there is a centre mode)

kappas = np.linspace(0, KAPPA_MAX, N_PTS)

# ── Truncated two-mode system ────────────────────────────────────────────────
M2 = np.array([scattering_matrix([k * np.exp(1j * PHASE)], N=2, gamma_i=GAMMA_I)
               for k in kappas])
T_bar = np.abs(M2[:, 0, 0]) ** 2
T_cross = np.abs(M2[:, 1, 0]) ** 2

# Self-check against the closed form (lossless only)
if GAMMA_I == 0:
    assert np.allclose(T_cross, 4 * kappas**2 / (1 + kappas**2) ** 2)
    assert max(unitarity_error(M) for M in M2) < 1e-12

k50 = kappas[np.argmin(np.abs(T_cross - 0.5))]
print(f"50:50 at kappa ≈ {k50:.4f}  (analytic √2-1 = {np.sqrt(2) - 1:.4f})")
print(f"Cross at kappa_max = {KAPPA_MAX}: {T_cross[-1]:.4f}")

# ── Untruncated comb: input in centre mode ───────────────────────────────────
c = N_WIDE // 2
MW = np.array([scattering_matrix([k * np.exp(1j * PHASE)], N=N_WIDE, gamma_i=GAMMA_I)
               for k in kappas])
P = np.abs(MW[:, :, c]) ** 2          # power in each output mode, shape (N_PTS, N_WIDE)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

ax[0].plot(kappas, T_bar, label=r"bar $|M_{00}|^2$")
ax[0].plot(kappas, T_cross, label=r"cross $|M_{10}|^2$")
ax[0].plot(kappas, 1 - T_bar - T_cross, "k:", label="loss")
ax[0].axvline(np.sqrt(2) - 1, color="grey", ls="--", lw=0.8)
ax[0].set(title="N = 2 (truncated)", xlabel=r"$|\kappa_1|/\gamma_e$", ylabel="Power")
ax[0].legend()

for m in range(N_WIDE):
    ls = "-" if abs(m - c) <= 1 else "--"
    ax[1].plot(kappas, P[:, m], ls, label=f"m = {m - c:+d}")
ax[1].set(title=f"N = {N_WIDE} (input at m = 0)", xlabel=r"$|\kappa_1|/\gamma_e$")
ax[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/single_tone_splitting_ratio.png", dpi=120)
plt.show()

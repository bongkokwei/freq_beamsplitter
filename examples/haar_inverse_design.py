"""
examples/haar_inverse_design.py
================================
Demonstrates inverse design of κ_l parameters to implement a target
unitary (Haar-random, DFT / N-dim Hadamard, or a Pauli X/Y/Z on the
central mode pair) on the
cascaded ring-resonator platform.

Follows Buddhiraju et al. (2021) — gradient-based (L-BFGS) optimisation
of the fidelity:

    F(U, V) = |⟨U, V⟩| / (‖U‖_F ‖V‖_F)

System sizing from N (N = 5 reproduces the paper's 5×5 demonstration):
    N_f = N - 1                    tones l = 1..N-1 reach every mode pair
    N_r = ceil(N² / (2·N_f))       ≈ (N + 1) / 2 for large N
    N = 5  ->  N_f = 4, N_r = 4

DOF check:
    Required  : N²          = 25 real DOF  (5×5 unitary)
    Available : 2·N_r·N_f   = 32 real DOF  ✓
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from freq_beamsplitter import (
    haar_unitary,
    inverse_design,
    cascaded_scattering_matrix,
    unitarity_error,
    embed_unitary_2x2,
)

PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# ── 1. Config ────────────────────────────────────────────────────────────────

N = 5  # frequency modes (odd or even)
N_f = N - 1  # tones per ring: l = 1..N-1 spans every mode separation
N_r = int(np.ceil(N**2 / (2 * N_f)))  # DOF count: 2·N_r·N_f >= N²
GAMMA_I = 0.0  # intrinsic loss / gamma_e, used for design AND reconstruction
TARGET = "haar"  # "haar" | "dft" (N-dim Hadamard) | "X" | "Y" | "Z" (Pauli on central pair)
SEED = 7  # reproducible Haar sample

c = N // 2  # centre index: sideband label m = k - c

# ── 2. Target unitary ────────────────────────────────────────────────────────

if TARGET == "haar":
    U_target = haar_unitary(N, seed=SEED)
elif TARGET == "dft":  # F_jk = ω^{jk}/√N, ω = e^{2πi/N}; equals Hadamard at N = 2
    U_target = np.exp(2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
elif TARGET in PAULI:
    k = (N - 1) // 2  # central pair (k, k+1); N = 2 -> the Pauli itself
    U_target = embed_unitary_2x2(PAULI[TARGET], k, k + 1, N)
else:
    raise ValueError(f"unknown TARGET {TARGET!r}")

print("=" * 60)
print(f"Inverse design demo — target: {TARGET}")
print("=" * 60)
print(f"Target U  :  {N}×{N}")
print(f"System    :  N_r={N_r} rings, N_f={N_f} tones/ring, N={N}, γ_i={GAMMA_I}")
print(f"Unitarity check on U: {unitarity_error(U_target):.2e}  (expect ~1e-15)\n")

# ── 3. Run inverse design ─────────────────────────────────────────────────────

result = inverse_design(
    U_target,
    N=N,
    N_r=N_r,
    N_f=N_f,
    gamma_e=1,
    gamma_i=GAMMA_I,
    delta_omega_list=0.0,
    n_restarts=5,
    kappa_scale=0.5,
    fidelity_tol=1 - 1e-4,
    seed=SEED,
    verbose=True,
)

# ── 4. Reconstruct and inspect the optimised matrix ──────────────────────────

M_opt = cascaded_scattering_matrix(result.kappa_list, N, gamma_i=GAMMA_I)

print(f"\nOptimised system")
print(f"  Fidelity F(U, M)     : {result.fidelity:.6f}")
print(f"  Unitarity error ‖M†M-I‖_F : {unitarity_error(M_opt):.2e}")
print(f"  Converged            : {result.converged}")

# ── 5. Print κ_l table (matching Supp. Note 6 of the paper) ──────────────────

print(f"\nOptimised κ_l / γ  (polar form):")
header = "  {:>8}".format("") + "".join(f"  Ring {r+1:>2}" for r in range(N_r))
print(header)
for l, kl_across_rings in enumerate(zip(*result.kappa_list), start=1):
    row = f"  κ_{l}/γ  "
    for kl in kl_across_rings:
        mag = np.abs(kl)
        phase = np.angle(kl) / np.pi
        row += f"  {mag:.4f}·e^i{phase:+.4f}π"
    print(row)

# ── 6. Visualisation ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 6a. Convergence curve ─────────────────────────────────────────────────────
ax_conv = fig.add_subplot(gs[0, 0])
if result.loss_history:
    iters = np.arange(1, len(result.loss_history) + 1)
    ax_conv.semilogy(
        iters, [1 - f for f in result.loss_history], lw=1.8, color="steelblue"
    )
    ax_conv.axhline(
        1 - result.fidelity,
        color="tomato",
        ls="--",
        lw=1.2,
        label=f"Final: 1-F = {1-result.fidelity:.2e}",
    )
    ax_conv.set_xlabel("L-BFGS iteration")
    ax_conv.set_ylabel("1 − F  (log scale)")
    ax_conv.set_title("Convergence (best restart)")
    ax_conv.legend(fontsize=8)
    ax_conv.grid(True, which="both", alpha=0.3)

# ── 6b. κ_l magnitudes per ring ───────────────────────────────────────────────
ax_kappa = fig.add_subplot(gs[1, 0])
x_pos = np.arange(N_f) + 1
width = 0.8 / N_r
for r, kl in enumerate(result.kappa_list):
    ax_kappa.bar(
        x_pos + r * width - 0.4 + width / 2,
        np.abs(kl),
        width=width,
        label=f"Ring {r+1}",
        alpha=0.85,
    )
ax_kappa.set_xlabel("Tone index l")
ax_kappa.set_ylabel("|κ_l| / γ")
ax_kappa.set_title("Optimised |κ_l| per ring")
ax_kappa.set_xticks(x_pos)
ax_kappa.legend(fontsize=8)
ax_kappa.grid(True, axis="y", alpha=0.3)

# ── 6c. κ_l phases per ring ───────────────────────────────────────────────────
ax_phase_kappa = fig.add_subplot(gs[0, 1], polar=True)
colours = plt.cm.tab10(np.linspace(0, 1, N_r))
for r, kl in enumerate(result.kappa_list):
    theta = np.angle(kl)
    r_val = np.abs(kl)
    ax_phase_kappa.scatter(
        theta, r_val, s=60, color=colours[r], label=f"Ring {r+1}", zorder=3
    )
    for t, rv in zip(theta, r_val):
        ax_phase_kappa.plot([0, t], [0, rv], color=colours[r], lw=1, alpha=0.5)
ax_phase_kappa.set_title("κ_l in complex plane (polar)", pad=14)
ax_phase_kappa.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))

# ── 6d–f. Scattering matrices: target, achieved, difference ──────────────────
lbls = [str(m) for m in range(-c, N - c)]


def _matrix_panel(ax, data, title, cmap, vmin, vmax, clabel):
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(N))
    ax.set_xticklabels(lbls, fontsize=7)
    ax.set_yticks(range(N))
    ax.set_yticklabels(lbls, fontsize=7)
    ax.set_xlabel("Input m'", fontsize=8)
    ax.set_ylabel("Output m", fontsize=8)


ax_tgt = fig.add_subplot(gs[0, 2])
ax_ach = fig.add_subplot(gs[1, 1])
ax_diff = fig.add_subplot(gs[1, 2])

_matrix_panel(ax_tgt, np.abs(U_target), "|U_target|", "viridis", 0, 1, "|·|")
_matrix_panel(ax_ach, np.abs(M_opt), "|M_achieved|", "viridis", 0, 1, "|·|")
_matrix_panel(
    ax_diff, np.abs(U_target - M_opt), "|U - M|  (error)", "hot", 0, 0.5, "|·|"
)

fig.suptitle(
    f"Inverse design ({TARGET})  —  "
    f"F = {result.fidelity:.5f}  "
    f"(N_r={N_r}, N_f={N_f}, N={N})",
    fontsize=11,
    y=1.01,
)

Path("figures").mkdir(exist_ok=True)
plt.savefig(
    f"figures/haar_inverse_design_Nr{N_r}_Nf{N_f}_N{N}.png",
    dpi=150,
    bbox_inches="tight",
)
print(f"\nFigure saved to haar_inverse_design_Nr{N_r}_Nf{N_f}_N{N}.png")
plt.show()

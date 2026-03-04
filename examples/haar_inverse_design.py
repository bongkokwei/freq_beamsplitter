"""
examples/haar_inverse_design.py
================================
Demonstrates inverse design of κ_l parameters to implement a Haar-random
unitary on the cascaded ring-resonator platform.

Follows Buddhiraju et al. (2021) — gradient-based (L-BFGS) optimisation
of the fidelity:

    F(U, V) = |⟨U, V⟩| / (‖U‖_F ‖V‖_F)

System parameters (matching the paper's 5×5 demonstration):
    N_sb = 2  →  5 frequency modes  {-2, -1, 0, +1, +2}
    N_f  = 4  modulation tones per ring
    N_r  = 4  rings in cascade

DOF check:
    Required  : (2·N_sb+1)² = 25 real DOF  (5×5 unitary)
    Available : 2·N_r·N_f   = 32 real DOF  ✓
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from freq_beamsplitter import (
    haar_unitary,
    inverse_design,
    cascaded_scattering_matrix,
    fidelity,
    unitarity_error,
    plot_matrix_grid,
    embed_unitary_2x2,
    beamsplitter,
)

# ── 1. System parameters ─────────────────────────────────────────────────────

N_sb = 3  # sidebands
N_f = 2 * N_sb  # modulation tones per ring
N_r = N_sb + 2  # rings in cascade
SEED = 7  # reproducible Haar sample

# ── 2. Draw a Haar-random target unitary ─────────────────────────────────────

U_target = haar_unitary(2 * N_sb + 1, seed=SEED)

# unitary_2x2 = beamsplitter(
#     theta=np.pi / 4, phi=0.0
# )  # 50:50 beamsplitter with phase shift
# U_target = embed_unitary_2x2(
#     U2=unitary_2x2,
#     i=0,
#     j=1,
#     N=2 * N_sb + 1,
# )  # embed in single input/output mode


print("=" * 60)
print("Haar-random unitary — inverse design demo")
print("=" * 60)
print(f"\nTarget U  :  {U_target.shape[0]}×{U_target.shape[1]} Haar-random unitary")
print(f"System    :  N_r={N_r} rings, N_f={N_f} tones/ring, N_sb={N_sb}")
print(f"Unitarity check on U: {unitarity_error(U_target):.2e}  (expect ~1e-15)\n")

# ── 3. Run inverse design ─────────────────────────────────────────────────────

result = inverse_design(
    U_target,
    N_sb=N_sb,
    N_r=N_r,
    N_f=N_f,
    gamma_e=0.8,
    gamma_i=0.1,
    delta_omega_list=0.0,
    n_restarts=5,
    kappa_scale=0.5,
    fidelity_tol=1 - 1e-4,
    seed=SEED,
    verbose=True,
)

# ── 4. Reconstruct and inspect the optimised matrix ──────────────────────────

M_opt = cascaded_scattering_matrix(result.kappa_list, N_sb)

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
N = 2 * N_sb + 1
lbls = [str(m) for m in range(-N_sb, N_sb + 1)]


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
    f"Haar-random unitary inverse design  —  "
    f"F = {result.fidelity:.5f}  "
    f"(N_r={N_r}, N_f={N_f}, N_sb={N_sb})",
    fontsize=11,
    y=1.01,
)

plt.savefig(
    f"figures/haar_inverse_design_Nr{N_r}_Nf{N_f}_Nsb{N_sb}.png",
    dpi=150,
    bbox_inches="tight",
)
print(f"\nFigure saved to haar_inverse_design_Nr{N_r}_Nf{N_f}_Nsb{N_sb}.png")
plt.show()
